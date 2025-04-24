#!/usr/bin/env python
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from datasets import Dataset, load_dataset

class LLMTrainer:
    """
    Trains a small LLM on pose analysis data.
    The model is progressively trained as new pose models are added.
    """
    
    def __init__(
        self,
        data_dir="llm_training_data",
        model_dir="llm_models",
        base_model="facebook/opt-350m",  # Small model for efficiency
        device="auto"
    ):
        """
        Initialize the LLM trainer.
        
        Args:
            data_dir (str): Directory with training data
            model_dir (str): Directory to save trained models
            base_model (str): Base model to fine-tune
            device (str): Device to use for training ('cpu', 'cuda', 'auto')
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.base_model = base_model
        
        # Determine the device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load config if it exists
        self.config_file = self.model_dir / "trainer_config.json"
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "base_model": base_model,
                "current_model": None,
                "training_runs": [],
                "total_epochs": 0,
                "last_update": None
            }
    
    def save_config(self):
        """Save the current configuration."""
        self.config["last_update"] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def prepare_training_data(self, input_file=None):
        """
        Prepare the training data for the LLM.
        
        Args:
            input_file (str, optional): Path to the input file with training examples.
                If None, use the combined training data file.
            
        Returns:
            Dataset: HuggingFace dataset ready for training
        """
        if input_file is None:
            input_file = self.data_dir / "processed" / "combined_training_data.json"
        
        print(f"Preparing training data from {input_file}")
        
        # Load the training examples
        try:
            with open(input_file, 'r') as f:
                examples = json.load(f)
        except:
            print(f"Error loading training examples from {input_file}")
            return None
        
        # Convert to the format expected by transformers
        formatted_examples = []
        for ex in examples:
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            output = ex.get("output", "")
            
            # Format for instruction tuning
            text = f"<s>[INST] {instruction}"
            if inp:
                text += f"\n{inp}"
            text += f" [/INST] {output}</s>"
            
            formatted_examples.append({"text": text})
        
        # Create a HuggingFace dataset
        dataset = Dataset.from_list(formatted_examples)
        
        print(f"Prepared dataset with {len(dataset)} examples")
        return dataset
    
    def setup_model_and_tokenizer(self, use_4bit=True, use_lora=True):
        """
        Set up the model and tokenizer for training.
        
        Args:
            use_4bit (bool): Whether to use 4-bit quantization
            use_lora (bool): Whether to use LoRA for parameter-efficient fine-tuning
            
        Returns:
            tuple: (model, tokenizer)
        """
        print(f"Setting up model and tokenizer based on {self.base_model}")
        
        # Set up quantization config if needed
        if use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # Ensure the pad token is set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
        # Load the model
        if self.config["current_model"] and Path(self.config["current_model"]).exists():
            # Load the latest model if it exists
            print(f"Loading previously trained model: {self.config['current_model']}")
            if use_lora:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    quantization_config=quantization_config,
                    device_map=self.device
                )
                model = PeftModel.from_pretrained(base_model, self.config["current_model"])
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config["current_model"],
                    quantization_config=quantization_config,
                    device_map=self.device
                )
        else:
            # Load the base model
            print(f"Loading base model: {self.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map=self.device
            )
            
            # Apply LoRA if needed
            if use_lora:
                if use_4bit:
                    model = prepare_model_for_kbit_training(model)
                
                # LoRA configuration
                lora_config = LoraConfig(
                    r=16,  # Rank
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
                        "gate_proj", "up_proj", "down_proj"  # MLP modules
                    ],
                )
                
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                
        return model, tokenizer
    
    def train_model(self, dataset, epochs=3, batch_size=4, learning_rate=5e-5, use_lora=True):
        """
        Train the model on the provided dataset.
        
        Args:
            dataset: HuggingFace dataset for training
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
            use_lora (bool): Whether to use LoRA for training
            
        Returns:
            str: Path to the trained model
        """
        print(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        # Set up the model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer(use_4bit=True, use_lora=use_lora)
        
        # Set up the data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create output directory for this run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.model_dir / f"run_{run_id}"
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            weight_decay=0.01,
            warmup_ratio=0.05,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch",
            report_to="tensorboard",
            push_to_hub=False
        )
        
        # Create the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        if use_lora:
            model.save_pretrained(output_dir / "peft_model")
            model_path = output_dir / "peft_model"
        else:
            trainer.save_model(output_dir / "full_model")
            model_path = output_dir / "full_model"
        
        # Save the tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Update the configuration
        self.config["current_model"] = str(model_path)
        self.config["total_epochs"] += epochs
        self.config["training_runs"].append({
            "run_id": run_id,
            "model_path": str(model_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "use_lora": use_lora,
            "dataset_size": len(dataset),
            "date": datetime.now().isoformat()
        })
        self.save_config()
        
        print(f"Training completed. Model saved to {model_path}")
        return str(model_path)
    
    def progressive_training(self, input_file=None, epochs=3, batch_size=4, learning_rate=5e-5, use_lora=True):
        """
        Perform progressive training on a new dataset.
        
        Args:
            input_file (str, optional): Path to the new training data.
                If None, use the combined training data file.
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
            use_lora (bool): Whether to use LoRA for training
            
        Returns:
            str: Path to the trained model
        """
        # Prepare the training data
        dataset = self.prepare_training_data(input_file)
        if dataset is None:
            print("No training data available. Aborting training.")
            return None
        
        # Train the model
        model_path = self.train_model(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_lora=use_lora
        )
        
        return model_path
    
    def generate_response(self, prompt, max_length=200):
        """
        Generate a response from the trained model.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum length of the generated response
            
        Returns:
            str: Generated response
        """
        # Check if we have a trained model
        if not self.config["current_model"] or not Path(self.config["current_model"]).exists():
            print("No trained model available. Please train a model first.")
            return None
        
        # Load the model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Format the prompt
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response
        response = response.split("[/INST]")[1].strip()
        # Remove the EOS token if present
        response = response.replace("</s>", "").strip()
        
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM on pose analysis data")
    parser.add_argument("--data_dir", default="llm_training_data", help="Directory with training data")
    parser.add_argument("--model_dir", default="llm_models", help="Directory to save trained models")
    parser.add_argument("--base_model", default="facebook/opt-350m", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--input_file", help="Path to specific training data file (optional)")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA training")
    parser.add_argument("--test", action="store_true", help="Test the model after training")
    
    args = parser.parse_args()
    
    # Create and run the trainer
    trainer = LLMTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        base_model=args.base_model
    )
    
    # Train the model
    model_path = trainer.progressive_training(
        input_file=args.input_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=not args.no_lora
    )
    
    # Test the model if requested
    if args.test and model_path:
        print("\nTesting the trained model:")
        test_prompts = [
            "Describe the body positioning in this video.",
            "What motion patterns can you identify in the analyzed video?",
            "Analyze the posture of the person in this video.",
            "What can you tell me about the movement technique based on the pose data?"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = trainer.generate_response(prompt)
            print(f"Response: {response}") 