# 1. First set up the LLM training with the datasets
python run_pipeline.sh setup-llm \
  --cud-dataset datasets/CUD/ \
  --maxplanck-dataset datasets/MAXPLANCK/ \
  --nomo-dataset datasets/NOMO/ \
  --base-model facebook/opt-1.3b

# 2. Then start the training
python run_pipeline.py train-llm

python scripts/generate_visualizations.py --data-dir public/results --biomechanics-only