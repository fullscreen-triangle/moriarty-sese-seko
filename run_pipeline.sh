#!/bin/bash

# Run the Moriarty distributed video processing pipeline

# Set up environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_EXEC="$(pwd)/.venv/bin/python"
    echo "Using PyCharm .venv Python: $PYTHON_EXEC"
elif [ -d "venv" ]; then
    source venv/bin/activate
    PYTHON_EXEC="$(pwd)/venv/bin/python"
    echo "Using venv Python: $PYTHON_EXEC"
elif [ -d "env" ]; then
    source env/bin/activate
    PYTHON_EXEC="$(pwd)/env/bin/python"
    echo "Using env Python: $PYTHON_EXEC"
else
    PYTHON_EXEC="python3"
    echo "No virtual environment found, using system Python3: $PYTHON_EXEC"
fi

# Make sure the current directory is in PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "Using Python path: ${PYTHONPATH}"
echo "Running pipeline..."

# Process videos and train LLM
"$PYTHON_EXEC" run_pipeline.py \
  --memory_limit 0.4 \
  --batch_size 30 \
  --input public \
  --output output \
  --models models \
  --llm_data llm_training_data \
  --llm_models llm_models \
  --train_llm \
  --both_llms

# To process a single video, uncomment and modify this:
# "$PYTHON_EXEC" run_pipeline.py --video public/your_video.mp4 --memory_limit 0.4 --both_llms

echo "Pipeline execution complete!" 