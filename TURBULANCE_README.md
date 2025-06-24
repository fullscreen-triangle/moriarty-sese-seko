# Turbulance Compiler

**A domain-specific language compiler for advanced sports analysis and evidence-based reasoning**

Turbulance is a specialized programming language designed to transform sports video analysis and biomechanical research from haphazard data processing into structured, evidence-based, probabilistic reasoning systems. It integrates seamlessly with the Moriarty framework to provide a comprehensive solution for sports science research and performance optimization.

## 🚀 Key Features

### Language Constructs
- **Propositions**: Structured hypothesis testing framework
- **Motions**: Sub-hypotheses within propositions for granular analysis
- **Evidence**: Multi-modal data integration and validation
- **Bayesian Networks**: Probabilistic reasoning with fuzzy logic updates
- **Sensor Fusion**: Multi-sensor data integration with uncertainty quantification
- **Fuzzy Systems**: Handle uncertainty and imprecision in measurements
- **Real-time Analysis**: Low-latency streaming analysis pipelines
- **Optimization Frameworks**: Genetic algorithms and multi-objective optimization
- **Metacognitive Analysis**: Self-improving system capabilities

### Advanced Analysis Capabilities
- **Biomechanical Analysis**: Joint angles, force estimation, movement patterns
- **Performance Optimization**: Technique improvement and strategy optimization
- **Temporal Analysis**: Time-series pattern recognition and phase analysis
- **Causal Inference**: Understanding cause-effect relationships in performance
- **Pattern Matching**: Automated recognition of technique patterns
- **Uncertainty Quantification**: Confidence intervals and measurement reliability

## 🏗️ Architecture

The Turbulance compiler consists of several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Parser      │───▶│  Semantic       │───▶│   Code          │
│   (Pest Grammar)│    │  Analyzer       │    │  Generator      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Runtime      │◀───│   Moriarty      │◀───│   Generated     │
│   Environment   │    │  Integration    │    │  Python Code    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Building from Source

```bash
# Clone the repository (assuming you're in the Moriarty project)
cd /path/to/moriarty-sese-seko

# Build the Turbulance compiler
cargo build --release

# Install the CLI tool
cargo install --path .
```

### Python Integration

```bash
# Build Python bindings
pip install maturin
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/turbulance_compiler-*.whl
```

## 🎯 Quick Start

### 1. Basic Example

Create a file `basic_analysis.tbn`:

```turbulance
// Basic sprint analysis
proposition SprintAnalysis:
    context video_data = load_video("sprint.mp4")
    
    motion OptimalTechnique("Athlete demonstrates good technique"):
        stride_frequency_range: 4.5..5.2
        ground_contact_time_range: 0.08..0.12
    
    within video_data:
        given stride_frequency in OptimalTechnique.stride_frequency_range:
            support OptimalTechnique with_confidence(0.9)
```

### 2. Compile to Python

```bash
# Compile the Turbulance program
turbulance compile basic_analysis.tbn -o output/

# Run the generated Python code
python output/basic_analysis.py
```

### 3. Advanced Analysis

See `examples/sprint_analysis.tbn` for a comprehensive example that demonstrates:
- Multi-modal sensor fusion
- Bayesian network reasoning
- Fuzzy logic systems
- Real-time analysis pipelines
- Genetic optimization
- Metacognitive adaptation

## 🖥️ CLI Usage

### Compilation

```bash
# Compile a single file
turbulance compile analysis.tbn

# Compile multiple files
turbulance compile *.tbn -o output/

# Compile with debug information
turbulance compile analysis.tbn --debug

# Compile to different targets
turbulance compile analysis.tbn --target rust
turbulance compile analysis.tbn --target javascript
```

### Validation

```bash
# Validate syntax and semantics
turbulance validate analysis.tbn

# Detailed validation output
turbulance validate analysis.tbn --detailed
```

### Code Analysis

```bash
# Analyze code complexity
turbulance analyze analysis.tbn

# Generate analysis report
turbulance analyze analysis.tbn --report
```

### Formatting

```bash
# Format code (print to stdout)
turbulance format analysis.tbn

# Format in place
turbulance format analysis.tbn --in-place
```

### Documentation

```bash
# Generate HTML documentation
turbulance doc analysis.tbn --format html

# Generate Markdown documentation
turbulance doc analysis.tbn --format markdown
```

### Project Creation

```bash
# Create a new basic project
turbulance new my_analysis

# Create a sports analysis project
turbulance new sprint_study --template sports

# Create a research project
turbulance new biomech_research --template research
```

## 🔧 API Usage

### Rust API

```rust
use turbulance_compiler::{TurbulanceCompiler, CompilationTarget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create compiler
    let compiler = TurbulanceCompiler::new("./output".to_string())
        .with_target(CompilationTarget::Python)
        .with_debug(true);
    
    // Compile source code
    let result = compiler.compile_string(
        r#"
        proposition TestAnalysis:
            motion TestMotion("Test description"):
                threshold: 0.8
        "#,
        "test_program"
    )?;
    
    println!("Generated code:\n{}", result.source_code);
    println!("Compilation time: {:.2}ms", result.metrics.total_time_ms);
    
    Ok(())
}
```

### Python API

```python
import turbulance_compiler

# Create compiler instance
compiler = turbulance_compiler.PyTurbulanceCompiler("./output")

# Compile Turbulance code
source_code = """
proposition SprintAnalysis:
    motion OptimalTechnique("Good technique"):
        threshold: 0.9
"""

generated_python = compiler.compile_string(source_code)
print("Generated Python code:")
print(generated_python)

# Validate code
is_valid = compiler.validate_source(source_code)
print(f"Code is valid: {is_valid}")
```

## 📚 Language Reference

### Core Constructs

#### Propositions
Propositions represent hypotheses that can be tested with evidence:

```turbulance
proposition HypothesisName:
    context variable_name = expression
    
    motion SubHypothesis("Description"):
        property1: value1
        property2: value2
    
    within data_source:
        // Analysis logic
```

#### Evidence
Evidence blocks define data sources and validation criteria:

```turbulance
evidence DataSource:
    sources:
        - sensor1: SensorType(properties...)
        - sensor2: SensorType(properties...)
    
    validation_criteria:
        accuracy: threshold_value
        reliability: threshold_value
```

#### Bayesian Networks
Define probabilistic relationships between variables:

```turbulance
bayesian_network NetworkName:
    nodes:
        - node1: NodeType(properties...)
        - node2: NodeType(properties...)
    
    edges:
        - node1 -> node2: relationship_strength(value, fuzziness: value)
```

#### Fuzzy Systems
Handle uncertainty and imprecision:

```turbulance
fuzzy_system SystemName:
    membership_functions:
        variable_name:
            - linguistic_label: function_type(parameters...)
    
    fuzzy_rules:
        - description: "Rule description"
          conditions: [(variable, label), ...]
          conclusions: [(variable, action, value), ...]
          weight: value
```

## 🔗 Integration with Moriarty

Turbulance seamlessly integrates with the Moriarty framework:

```turbulance
// Automatic integration with Moriarty components
item pose_data = analyze_pose(video_data)
item movement_patterns = track_movement(pose_data)
item biomechanics = analyze_biomechanics(movement_patterns)

// Use Moriarty's processing pipeline
within moriarty_pipeline_results:
    biomechanical joint_analysis:
        // Turbulance analysis logic using Moriarty data
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Test specific module
cargo test turbulance::parser

# Test Python bindings
python -m pytest tests/
```

## 📈 Performance

The Turbulance compiler is designed for performance:

- **Parse Time**: < 10ms for typical programs
- **Code Generation**: < 50ms for complex analyses
- **Memory Usage**: < 100MB for large programs
- **Runtime Performance**: Near-native speed through optimized Python generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
cargo install cargo-watch
pip install maturin pytest

# Run tests on file changes
cargo watch -x test

# Format code
cargo fmt

# Check for issues
cargo clippy
```

## 🐛 Troubleshooting

### Common Issues

1. **Parse Errors**: Check syntax against examples in `examples/`
2. **Python Integration Issues**: Ensure PyO3 and maturin are properly installed
3. **Performance Issues**: Use `--debug` flag to identify bottlenecks

### Debug Mode

```bash
# Enable debug output
turbulance compile analysis.tbn --debug --verbose

# Check AST generation
RUST_LOG=debug turbulance compile analysis.tbn
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the [Moriarty](README.md) sports analysis framework
- Uses [Pest](https://pest.rs/) for parsing
- Integrates with [PyO3](https://pyo3.rs/) for Python bindings
- Inspired by domain-specific languages in scientific computing

## 📞 Support

- **Documentation**: See the `docs/language/` directory
- **Examples**: Check the `examples/` directory
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

---

**Turbulance: Transforming sports analysis through structured reasoning and evidence-based computation.** 