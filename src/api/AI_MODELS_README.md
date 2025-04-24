# AI Models for Biomechanical Analysis

This module provides specialized AI models for biomechanical analysis through Hugging Face's Inference API. Instead of using generic commercial models like OpenAI or Anthropic directly, this implementation allows using more specialized models for different aspects of biomechanical analysis.

## Setup

1. **Hugging Face API Token**:
   - Create an account on [Hugging Face](https://huggingface.co/)
   - Generate an API token from your profile settings
   - Set the token as an environment variable: `HF_API_TOKEN`
   - Or provide it directly when initializing the client

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Using the MoriartyLLMClient

The `MoriartyLLMClient` class provides a unified interface to access different specialized models:

```python
from src.api.ai_models import MoriartyLLMClient

# Initialize with API token
client = MoriartyLLMClient(api_token="your_hf_token")
# Or use environment variable
client = MoriartyLLMClient()

# Analyze biomechanical data
results = client.analyze_sync(
    analysis_type="biomechanical_analysis",
    data=biomechanical_data,
    athlete_info={"sport": "sprint", "level": "elite"}
)

# For asynchronous use
import asyncio

async def analyze():
    results = await client.analyze(
        analysis_type="movement_patterns",
        data=pose_sequence_data,
        reference_patterns=reference_patterns
    )
    return results

results = asyncio.run(analyze())
```

## Available Analysis Types

The client supports different analysis types, each using a specialized model:

| Analysis Type | Model | Description |
|---------------|-------|-------------|
| `biomechanical_analysis` | `anthropic/claude-3-opus-20240229` | Comprehensive biomechanical analysis |
| `movement_patterns` | `meta-llama/llama-3-70b-instruct` | Movement pattern comparison against references |
| `technical_reporting` | `anthropic/claude-3-haiku-20240307` | Technical report generation |
| `sprint_specialist` | `your-username/sprint-biomechanics-expert` | Sprint-specific technique analysis |
| `performance_comparison` | `your-username/performance-comparison-model` | Performance comparison over time |
| `coaching_insights` | `anthropic/claude-3-sonnet-20240229` | Elite coaching insights |
| `quick_analysis` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Fast, focused analysis |

## Standalone Functions

For simpler use cases, you can use the standalone functions:

```python
from src.api.ai_models import analyze_biomechanics, analyze_movement_patterns, generate_technical_report

# Analyze biomechanics
results = analyze_biomechanics(
    biomechanical_data=data,
    athlete_info={"sport": "sprint"}
)

# Analyze movement patterns
results = analyze_movement_patterns(
    pose_sequence_data=pose_data,
    reference_patterns=ref_patterns
)

# Generate technical report
report = generate_technical_report(
    analysis_results=analysis_data,
    athlete_profile=profile
)
```

## Using with AI Dynamics Analyzer

The `AIDynamicsAnalyzer` class extends the base `DynamicsAnalyzer` with AI capabilities:

```python
from src.core.dynamics.ai_dynamics_analyzer import AIDynamicsAnalyzer

# Initialize
analyzer = AIDynamicsAnalyzer()

# Analyze biomechanics
results = analyzer.analyze_biomechanics(
    positions_batch=positions,
    velocities_batch=velocities,
    accelerations_batch=accelerations,
    athlete_info={"sport": "sprint"}
)

# Generate technical report
report = analyzer.generate_technical_report(results)
```

## Example

See `src/examples/ai_biomechanics_example.py` for a complete example of using the AI models.

```bash
# Run the example
python src/examples/ai_biomechanics_example.py --api-token YOUR_TOKEN
```

## Notes on Fine-tuning

For optimal performance, you can fine-tune specialized models on biomechanical data:

1. Use Hugging Face to fine-tune models on your specific data
2. Replace the model IDs in the client with your fine-tuned models
3. For sprint-specific and performance comparison, replace the placeholders with your actual fine-tuned models

## Caching

Results are cached based on input data to avoid redundant API calls. The cache is maintained for the lifetime of the client instance.

## Error Handling

API calls are wrapped in try-except blocks and errors are logged. For production use, implement appropriate error handling and rate limiting strategies. 