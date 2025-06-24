# Installation Guide

This guide provides multiple ways to install the Moriarty package.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- CUDA-capable GPU (recommended for optimal performance)
- API keys for OpenAI/Anthropic (optional, for AI analysis)

## Option 1: Install from Source

This is the recommended approach during development:

```bash
# Clone the repository
git clone https://github.com/yourusername/moriarty.git
cd moriarty

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Option 2: Install Using setup.sh Script

For convenience, you can use the provided setup script:

```bash
# On Unix/Linux/Mac:
./setup.sh

# On Windows:
setup.bat
```

## Option 3: Install Dependencies Separately

If you prefer to install dependencies manually:

```bash
# Install main dependencies
pip install -r requirements.txt

# For API functionality
pip install -r src/api/requirements_api.txt  # If available

# For LLM functionality
pip install -r requirements_llm.txt  # If available
```

## Environment Setup

For API integration with OpenAI or Claude, create a `.env` file:

```bash
# Create a .env file in the project root
touch .env  # On Windows: type nul > .env

# Add your API keys
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

## Verifying Installation

To verify that the installation was successful:

```bash
# Check if the package can be imported
python -c "import moriarty; print(moriarty.__version__)"

# Run a simple test
moriarty --help
```

## Troubleshooting

If you encounter any issues during installation:

- Make sure you have the correct Python version (3.8+)
- Check that all dependencies are properly installed
- For GPU acceleration, ensure CUDA is properly set up
- For API functionality, verify that API keys are correctly set in .env

For more detailed information, refer to the main documentation in the `docs/` directory. 