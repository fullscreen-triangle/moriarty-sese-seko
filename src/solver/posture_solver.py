import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("posture_solver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("posture_solver")

# Constants
MIN_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL_REGISTRY = "models/posture"


class ModelLoadError(Exception):
    """Exception raised when a model fails to load properly"""
    pass


class AnalysisError(Exception):
    """Exception raised when posture analysis fails"""
    pass


class PostureModel:
    """Base class for posture analysis models"""
    
    def __init__(self, model_path: str):
        """
        Initialize a posture model
        
        Args:
            model_path: Path to the model files
        
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        self.model_path = model_path
        logger.info(f"Initializing {self.__class__.__name__} from {model_path}")
        try:
            self.model = self._load_model()
            logger.info(f"Successfully loaded {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ModelLoadError(f"Failed to load {self.__class__.__name__}: {str(e)}")
    
    def _load_model(self):
        """Load model from path - would be implemented by specific model types"""
        raise NotImplementedError("_load_model must be implemented by subclasses")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction using model
        
        Args:
            features: Dictionary of features extracted from posture data
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            AnalysisError: If prediction fails
        """
        raise NotImplementedError("predict must be implemented by subclasses")
    
    def validate_input(self, features: Dict[str, Any]) -> bool:
        """
        Validate that the input features contain required data
        
        Args:
            features: Dictionary of features to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        raise NotImplementedError("validate_input must be implemented by subclasses")


class SpineAlignmentModel(PostureModel):
    """Model to analyze spine alignment and identify deviations"""
    
    def _load_model(self):
        """
        Load spine alignment model
        
        In production, this would load a trained model from disk/cloud
        For now, we're using a simple rule-based model
        
        Returns:
            Loaded model object
        """
        logger.debug(f"Loading spine alignment model from {self.model_path}")
        
        # Check if path exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path {self.model_path} does not exist, using default parameters")
        
        # In production, this would load model weights, parameters, etc.
        # For demonstration, we return configuration parameters
        return {
            "name": "spine_alignment",
            "version": "1.0",
            "thresholds": {
                "excessive_straightness": 170,
                "severe_straightness": 175,
                "excessive_curvature": 150,
                "severe_curvature": 140
            }
        }
    
    def validate_input(self, features: Dict[str, Any]) -> bool:
        """
        Validate spine keypoints exist and are sufficient for analysis
        
        Args:
            features: Dictionary of features to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        if "keypoints" not in features:
            logger.warning("No keypoints found in features")
            return False
            
        spine_keypoints = features.get("keypoints", {}).get("spine", [])
        if not spine_keypoints or len(spine_keypoints) < 3:
            logger.warning(f"Insufficient spine keypoints: {len(spine_keypoints) if spine_keypoints else 0}")
            return False
            
        # Validate keypoint format
        try:
            for point in spine_keypoints:
                if len(point) != 2 or not all(isinstance(x, (int, float)) for x in point):
                    logger.warning(f"Invalid keypoint format: {point}")
                    return False
        except Exception as e:
            logger.warning(f"Error validating keypoints: {str(e)}")
            return False
            
        return True
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze spine alignment from keypoints
        
        Args:
            features: Dictionary containing spine keypoints
            
        Returns:
            Dictionary with alignment analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        # Validate input
        if not self.validate_input(features):
            raise AnalysisError("Invalid input for spine alignment analysis")
        
        try:
            # Extract relevant joint positions from features
            spine_keypoints = features.get("keypoints", {}).get("spine", [])
            
            # Simple analysis for demonstration
            # Calculate spine curvature by measuring angles between keypoints
            angles = []
            for i in range(len(spine_keypoints) - 2):
                p1 = np.array(spine_keypoints[i])
                p2 = np.array(spine_keypoints[i+1])
                p3 = np.array(spine_keypoints[i+2])
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle in degrees using the dot product
                cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
            
            # Analyze angles to determine alignment issues
            avg_angle = np.mean(angles)
            deviation = np.std(angles)
            
            # Classification based on average and deviation using model thresholds
            thresholds = self.model["thresholds"]
            
            if avg_angle > thresholds["excessive_straightness"]:
                alignment = "excessive_straightness"
                severity = "moderate" if avg_angle <= thresholds["severe_straightness"] else "severe"
            elif avg_angle < thresholds["excessive_curvature"]:
                alignment = "excessive_curvature"
                severity = "severe" if avg_angle < thresholds["severe_curvature"] else "moderate"
            else:
                alignment = "normal"
                severity = "none"
                
            # Calculate confidence based on consistency of measurements
            # Lower standard deviation means more confident measurement
            confidence = max(0.5, min(1.0, 1.0 - (deviation / 45.0)))
            
            return {
                "alignment": alignment,
                "severity": severity,
                "avg_angle": float(avg_angle),
                "deviation": float(deviation),
                "raw_angles": [float(a) for a in angles],
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Error in spine alignment analysis: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisError(f"Spine alignment analysis failed: {str(e)}")


class ShoulderBalanceModel(PostureModel):
    """Model to analyze shoulder balance and symmetry"""
    
    def _load_model(self):
        """
        Load shoulder balance model
        
        In production, this would load a trained model from disk/cloud
        For now, we're using a simple rule-based model
        
        Returns:
            Loaded model object
        """
        logger.debug(f"Loading shoulder balance model from {self.model_path}")
        
        # Check if path exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path {self.model_path} does not exist, using default parameters")
        
        # In production, this would load model weights, parameters, etc.
        return {
            "name": "shoulder_balance",
            "version": "1.0",
            "thresholds": {
                "balanced": 0.05,
                "slight_imbalance": 0.10,
                "moderate_imbalance": 0.15
            }
        }
    
    def validate_input(self, features: Dict[str, Any]) -> bool:
        """
        Validate shoulder keypoints exist and are sufficient for analysis
        
        Args:
            features: Dictionary of features to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        if "keypoints" not in features:
            logger.warning("No keypoints found in features")
            return False
            
        shoulder_left = features.get("keypoints", {}).get("shoulder_left")
        shoulder_right = features.get("keypoints", {}).get("shoulder_right")
        
        if not shoulder_left or not shoulder_right:
            logger.warning("Missing shoulder keypoints")
            return False
            
        # Validate keypoint format
        try:
            for point in [shoulder_left, shoulder_right]:
                if len(point) != 2 or not all(isinstance(x, (int, float)) for x in point):
                    logger.warning(f"Invalid shoulder keypoint format: {point}")
                    return False
        except Exception as e:
            logger.warning(f"Error validating shoulder keypoints: {str(e)}")
            return False
            
        return True
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze shoulder balance from keypoints
        
        Args:
            features: Dictionary containing shoulder keypoints
            
        Returns:
            Dictionary with shoulder balance analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        # Validate input
        if not self.validate_input(features):
            raise AnalysisError("Invalid input for shoulder balance analysis")
        
        try:
            shoulder_left = features.get("keypoints", {}).get("shoulder_left")
            shoulder_right = features.get("keypoints", {}).get("shoulder_right")
            
            # Calculate height difference
            height_diff = abs(shoulder_left[1] - shoulder_right[1])
            
            # Normalize by torso length for scale invariance
            hip_center = features.get("keypoints", {}).get("hip_center", [0, 0])
            shoulder_center = [(shoulder_left[0] + shoulder_right[0])/2, 
                              (shoulder_left[1] + shoulder_right[1])/2]
            torso_length = np.linalg.norm(np.array(shoulder_center) - np.array(hip_center))
            
            if torso_length > 0:
                normalized_diff = height_diff / torso_length
            else:
                logger.warning("Zero torso length detected, using unnormalized difference")
                normalized_diff = height_diff / 100.0  # Fallback normalization
            
            # Determine balance using model thresholds
            thresholds = self.model["thresholds"]
            
            if normalized_diff < thresholds["balanced"]:
                balance = "balanced"
                severity = "none"
            elif normalized_diff < thresholds["slight_imbalance"]:
                balance = "slight_imbalance"
                severity = "mild"
            elif normalized_diff < thresholds["moderate_imbalance"]:
                balance = "moderate_imbalance"
                severity = "moderate"
            else:
                balance = "significant_imbalance"
                severity = "severe"
                
            # Determine which shoulder is higher
            higher_side = "left" if shoulder_left[1] < shoulder_right[1] else "right"
            
            # Calculate confidence
            # More normalized difference = lower confidence
            confidence = max(0.7, min(0.95, 1.0 - normalized_diff))
            
            return {
                "balance": balance,
                "severity": severity,
                "higher_side": higher_side,
                "normalized_difference": float(normalized_diff),
                "absolute_difference_px": float(height_diff),
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Error in shoulder balance analysis: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisError(f"Shoulder balance analysis failed: {str(e)}")


class PostureAnalysisSolver:
    """
    Solver for posture analysis that uses specialized models to analyze posture data
    and generate detailed analysis with biomechanical reasoning.
    """
    
    def __init__(self, model_registry_path: Optional[str] = None):
        """
        Initialize the PostureAnalysisSolver with models from the registry
        
        Args:
            model_registry_path: Path to the directory containing model files.
                                If None, uses the default registry path.
        
        Raises:
            ModelLoadError: If models cannot be loaded
        """
        self.model_registry_path = model_registry_path or DEFAULT_MODEL_REGISTRY
        logger.info(f"Initializing PostureAnalysisSolver with registry: {self.model_registry_path}")
        
        try:
            self.models = self._load_models()
            if not self.models:
                logger.warning("No posture analysis models were loaded")
        except Exception as e:
            logger.error(f"Failed to initialize PostureAnalysisSolver: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ModelLoadError(f"Failed to initialize solver: {str(e)}")
    
    def _load_models(self) -> Dict[str, PostureModel]:
        """
        Load all available posture analysis models
        
        Returns:
            Dictionary of loaded models
            
        Raises:
            ModelLoadError: If a model fails to load
        """
        models = {}
        
        # Create model registry directory if it doesn't exist
        os.makedirs(self.model_registry_path, exist_ok=True)
        
        # Create model paths
        spine_model_path = os.path.join(self.model_registry_path, "spine_alignment")
        shoulder_model_path = os.path.join(self.model_registry_path, "shoulder_balance")
        
        # Initialize models
        try:
            logger.info("Loading spine alignment model...")
            models["spine_alignment"] = SpineAlignmentModel(spine_model_path)
        except Exception as e:
            logger.error(f"Failed to load spine alignment model: {str(e)}")
            logger.debug(traceback.format_exc())
        
        try:
            logger.info("Loading shoulder balance model...")
            models["shoulder_balance"] = ShoulderBalanceModel(shoulder_model_path)
        except Exception as e:
            logger.error(f"Failed to load shoulder balance model: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # In production, you would dynamically discover and load models
        # based on what's available in the registry
        
        logger.info(f"Loaded {len(models)} posture analysis models")
        return models
    
    def identify_analysis_type(self, query: str) -> List[str]:
        """
        Identify what types of posture analysis are needed based on the query
        
        Args:
            query: The user's question about posture
            
        Returns:
            List of analysis types to perform
        """
        analysis_types = []
        
        # Simple keyword matching - in production would use NLP techniques
        query = query.lower()
        
        logger.debug(f"Identifying analysis types for query: {query}")
        
        if any(word in query for word in ["spine", "back", "slouch", "hunch", "straight"]):
            analysis_types.append("spine_alignment")
            
        if any(word in query for word in ["shoulder", "balance", "even", "level", "tilt"]):
            analysis_types.append("shoulder_balance")
            
        # If nothing specific is mentioned, do a complete analysis
        if not analysis_types:
            analysis_types = list(self.models.keys())
            
        logger.debug(f"Identified analysis types: {analysis_types}")
        return analysis_types
    
    def extract_posture_features(self, posture_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant features from posture data
        
        Args:
            posture_data: Raw posture data (keypoints, angles, etc.)
            
        Returns:
            Dictionary of processed features for analysis
        """
        logger.debug("Extracting features from posture data")
        
        # In production this would process raw data into standardized format
        features = {
            "keypoints": {},
            "angles": {},
            "metadata": {}
        }
        
        # Copy relevant data
        if "keypoints" in posture_data:
            features["keypoints"] = posture_data["keypoints"]
            
        if "angles" in posture_data:
            features["angles"] = posture_data["angles"]
            
        if "metadata" in posture_data:
            features["metadata"] = posture_data["metadata"]
        
        # In production, you'd add normalization, standardization, and 
        # other preprocessing steps here
        
        # Validate the features contain required data
        if not features["keypoints"]:
            logger.warning("No keypoints found in posture data")
        
        return features
    
    def select_models(self, analysis_types: List[str], required_models: Optional[List[str]] = None) -> List[str]:
        """
        Select which models to use based on analysis types and requirements
        
        Args:
            analysis_types: Types of analysis to perform
            required_models: Specific models requested
            
        Returns:
            List of model names to use
        """
        if required_models:
            # Filter to only include models we have
            available_models = [model for model in required_models if model in self.models]
            logger.debug(f"Using required models (filtered by availability): {available_models}")
            return available_models
        
        # Select models based on analysis types
        available_models = [model_name for model_name in analysis_types if model_name in self.models]
        logger.debug(f"Selected models based on analysis types: {available_models}")
        return available_models
    
    def execute_models(self, model_names: List[str], features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute selected models with the extracted features
        
        Args:
            model_names: Names of models to execute
            features: Processed features for analysis
            
        Returns:
            Dictionary of results from all models
        """
        results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                model = self.models[model_name]
                logger.info(f"Executing {model_name} model")
                
                try:
                    model_result = model.predict(features)
                    results[model_name] = model_result
                    logger.debug(f"{model_name} analysis complete: {model_result.get('confidence', 0):.2f} confidence")
                except Exception as e:
                    logger.error(f"Error executing {model_name} model: {str(e)}")
                    logger.debug(traceback.format_exc())
                    results[model_name] = {"error": str(e), "confidence": 0.0}
            else:
                logger.warning(f"Model {model_name} not found in registry")
                    
        return results
    
    def generate_solution_trace(self, model_names: List[str], features: Dict[str, Any], 
                               results: Dict[str, Any]) -> str:
        """
        Generate a detailed solution trace with biomechanical reasoning
        
        Args:
            model_names: Names of models used
            features: Input features used
            results: Results from model execution
            
        Returns:
            Detailed explanation of the analysis process and reasoning
        """
        logger.debug("Generating solution trace")
        trace_lines = ["## Posture Analysis Process\n"]
        
        # Add features summary
        trace_lines.append("### Input Data\n")
        
        keypoint_count = len(features.get("keypoints", {}))
        trace_lines.append(f"- Processed {keypoint_count} keypoint sets")
        
        if "metadata" in features and "image_dimensions" in features["metadata"]:
            dims = features["metadata"]["image_dimensions"]
            trace_lines.append(f"- Image dimensions: {dims[0]}x{dims[1]} pixels")
        
        # Generate analysis for each model
        trace_lines.append("\n### Analysis Steps\n")
        
        for model_name in model_names:
            if model_name not in results:
                continue
                
            result = results[model_name]
            trace_lines.append(f"#### {model_name.replace('_', ' ').title()} Analysis\n")
            
            # Check for errors
            if "error" in result:
                trace_lines.append(f"Error in analysis: {result['error']}")
                continue
            
            if model_name == "spine_alignment":
                trace_lines.append("1. Extracted spine keypoints from posture data")
                trace_lines.append("2. Calculated angles between adjacent vertebral segments")
                
                if "raw_angles" in result:
                    angle_str = ", ".join([f"{angle:.1f}°" for angle in result["raw_angles"]])
                    trace_lines.append(f"3. Measured angles: {angle_str}")
                    
                if "avg_angle" in result:
                    trace_lines.append(f"4. Average angle: {result['avg_angle']:.1f}°")
                    
                if "alignment" in result:
                    trace_lines.append(f"5. Classification: {result['alignment'].replace('_', ' ')} " +
                                      f"(Severity: {result['severity']})")
                
            elif model_name == "shoulder_balance":
                trace_lines.append("1. Identified left and right shoulder positions")
                trace_lines.append("2. Measured vertical height difference between shoulders")
                
                if "absolute_difference_px" in result:
                    trace_lines.append(f"3. Absolute difference: {result['absolute_difference_px']:.1f} pixels")
                    
                if "normalized_difference" in result:
                    trace_lines.append(f"4. Normalized by torso length: {result['normalized_difference']:.3f}")
                    
                if "balance" in result and "higher_side" in result:
                    trace_lines.append(f"5. Classification: {result['balance'].replace('_', ' ')} with " +
                                      f"{result['higher_side']} shoulder higher (Severity: {result['severity']})")
        
        # Overall assessment
        trace_lines.append("\n### Overall Assessment\n")
        
        issues = []
        for model_name, result in results.items():
            if "error" in result:
                continue
                
            if model_name == "spine_alignment" and result.get("alignment") != "normal":
                issues.append(f"{result['severity']} {result['alignment'].replace('_', ' ')}")
                
            if model_name == "shoulder_balance" and result.get("balance") != "balanced":
                issues.append(f"{result['severity']} shoulder imbalance ({result['higher_side']} side higher)")
        
        if issues:
            trace_lines.append("Identified posture issues:")
            for issue in issues:
                trace_lines.append(f"- {issue}")
        else:
            trace_lines.append("No significant posture issues identified.")
        
        return "\n".join(trace_lines)
    
    def solve(self, query: str, posture_data: Dict[str, Any], 
             required_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Solve a posture analysis problem using appropriate models
        
        Args:
            query: The user's question about posture
            posture_data: Raw posture data (keypoints, angles, etc.)
            required_models: Specific models to use (optional)
            
        Returns:
            Dictionary containing analysis results and solution trace
            
        Raises:
            AnalysisError: If analysis fails
        """
        logger.info(f"Solving posture analysis for query: {query}")
        
        try:
            # Identify what type of posture analysis is needed
            analysis_types = self.identify_analysis_type(query)
            
            # Extract relevant features from posture data
            features = self.extract_posture_features(posture_data)
            
            # Select appropriate biomechanical models
            models_to_use = self.select_models(analysis_types, required_models)
            
            if not models_to_use:
                logger.warning("No suitable models found for the analysis")
                return {
                    "analysis_result": {},
                    "solution_method": "No suitable models found for analysis",
                    "models_used": [],
                    "analysis_types": analysis_types,
                    "confidence": 0.0
                }
            
            # Execute calculations
            results = self.execute_models(models_to_use, features)
            
            # Generate detailed analysis with biomechanical reasoning
            solution_trace = self.generate_solution_trace(models_to_use, features, results)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(results)
            
            logger.info(f"Analysis complete with {confidence:.2f} confidence")
            
            return {
                "analysis_result": results,
                "solution_method": solution_trace,
                "models_used": models_to_use,
                "analysis_types": analysis_types,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error solving posture analysis: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisError(f"Posture analysis failed: {str(e)}")
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate an overall confidence score based on model results
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Overall confidence score (0.0-1.0)
        """
        confidences = []
        
        for model_name, result in results.items():
            if "confidence" in result and isinstance(result["confidence"], (int, float)):
                confidences.append(result["confidence"])
                
        if not confidences:
            return 0.0
            
        # Weight confidences by model importance if needed
        # In production, you might have different weights for different models
        return sum(confidences) / len(confidences)


def create_distillation_trio(
    solver: PostureAnalysisSolver, 
    query: str, 
    posture_data: Dict[str, Any], 
    commercial_llm_client
) -> Dict[str, Any]:
    """
    Create a distillation trio (query-solution-answer) for knowledge distillation
    
    Args:
        solver: PostureAnalysisSolver instance
        query: User question about posture
        posture_data: Raw posture data for analysis
        commercial_llm_client: Client for commercial LLM API
        
    Returns:
        Dictionary containing query, solution method, answer and metadata
        
    Raises:
        AnalysisError: If analysis fails
        Exception: If LLM generation fails
    """
    logger.info(f"Creating distillation trio for query: {query}")
    
    try:
        # Get solver results with full biomechanical reasoning
        solver_output = solver.solve(query, posture_data)
        
        # Have commercial LLM interpret the technical results
        prompt = f"""
        Question: {query}
        
        Using the following biomechanical analysis method and results, provide a comprehensive answer:
        
        Method: {solver_output['solution_method']}
        Result: {json.dumps(solver_output['analysis_result'], indent=2)}
        
        Your answer should explain the biomechanical principles, interpret the results in context of proper posture,
        and provide relevant recommendations or insights.
        """
        
        logger.debug("Sending prompt to commercial LLM")
        llm_answer = commercial_llm_client.generate(prompt)
        
        if not llm_answer:
            logger.warning("Empty response received from LLM")
            llm_answer = "No response generated from language model."
        
        # Create a reference ID for the posture data
        # In production this would store the data in a database or file system
        posture_data_hash = hash(json.dumps(posture_data))
        posture_data_reference = f"posture_data_{posture_data_hash}"
        
        # Create trio
        trio = {
            "query": query,
            "posture_data_reference": posture_data_reference,
            "solution_method": solver_output['solution_method'],
            "answer": llm_answer,
            "metadata": {
                "models_used": solver_output.get("models_used", []),
                "analysis_types": solver_output.get("analysis_types", []),
                "confidence": solver_output.get("confidence", 0.0),
                "timestamp": Path(__file__).stat().st_mtime
            }
        }
        
        logger.info("Successfully created distillation trio")
        return trio
    except Exception as e:
        logger.error(f"Error creating distillation trio: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def generate_distillation_dataset(
    solver: PostureAnalysisSolver, 
    posture_data_samples: List[Dict[str, Any]],
    queries: List[str],
    commercial_llm_client,
    output_file: str
) -> None:
    """
    Generate a full distillation dataset by creating trios for multiple queries and posture samples
    
    Args:
        solver: PostureAnalysisSolver instance
        posture_data_samples: List of posture data samples
        queries: List of queries about posture
        commercial_llm_client: Client for commercial LLM API
        output_file: Path to output JSONL file
        
    Raises:
        Exception: If dataset generation fails
    """
    logger.info(f"Generating distillation dataset with {len(posture_data_samples)} posture samples and {len(queries)} queries")
    
    try:
        distillation_data = []
        success_count = 0
        error_count = 0
        
        for i, posture_data in enumerate(posture_data_samples):
            for j, query in enumerate(queries):
                try:
                    logger.debug(f"Processing sample {i+1}/{len(posture_data_samples)}, query {j+1}/{len(queries)}")
                    trio = create_distillation_trio(solver, query, posture_data, commercial_llm_client)
                    distillation_data.append(trio)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error creating trio for query '{query}': {str(e)}")
                    error_count += 1
        
        # Save distillation data as JSONL
        if distillation_data:
            with open(output_file, "w") as f:
                for item in distillation_data:
                    f.write(json.dumps(item) + "\n")
            
            logger.info(f"Generated {len(distillation_data)} distillation trios saved to {output_file}")
            logger.info(f"Results: {success_count} successful, {error_count} failed")
        else:
            logger.error("No distillation data was generated")
    except Exception as e:
        logger.error(f"Error generating distillation dataset: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# Example usage - commented out as this would be in a separate script
"""
# Initialize solver with model registry path
model_registry_path = "models/posture"
solver = PostureAnalysisSolver(model_registry_path)

# Mock LLM client
class MockLLMClient:
    def generate(self, prompt):
        return "Based on the biomechanical analysis, I can see that your posture shows a moderate spine curvature issue. The spine alignment model detected an average angle of 147.5 degrees between vertebral segments, which indicates excessive curvature. This is commonly associated with a slouching posture. Additionally, your shoulders show a slight imbalance with the right shoulder being 1.2cm higher than the left. I would recommend exercises to strengthen your core and upper back muscles, particularly focusing on the left side to help balance your shoulders. Regular stretching of chest muscles can also help improve your spine alignment."

llm_client = MockLLMClient()

# Example posture data
example_posture_data = {
    "keypoints": {
        "spine": [
            [300, 100], 
            [305, 150], 
            [315, 200], 
            [330, 250], 
            [350, 300]
        ],
        "shoulder_left": [250, 152],
        "shoulder_right": [360, 148],
        "hip_center": [320, 400]
    },
    "metadata": {
        "image_dimensions": [640, 480],
        "capture_conditions": "standing"
    }
}

# Generate a single trio
trio = create_distillation_trio(solver, 
                             "Is my back posture correct or am I slouching?", 
                             example_posture_data, 
                             llm_client)

print(json.dumps(trio, indent=2))

# Generate a dataset with multiple queries and posture samples
sample_queries = [
    "Is my back posture correct or am I slouching?",
    "Are my shoulders balanced?",
    "How is my overall posture?",
    "What posture issues do I have?",
    "Is my spine alignment healthy?"
]

# In production, this would be multiple real posture samples
posture_samples = [example_posture_data]

generate_distillation_dataset(solver, 
                             posture_samples, 
                             sample_queries, 
                             llm_client, 
                             "posture_distillation_data.jsonl")
""" 