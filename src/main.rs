mod turbulance;

use clap::{Parser, Subcommand};
use anyhow::{Result, anyhow};
use std::path::PathBuf;
use log::{info, warn, error};
use env_logger::Env;

use turbulance::{TurbulanceCompiler, CompilationTarget, utils};

/// Turbulance Compiler - A domain-specific language for sports analysis and evidence-based reasoning
#[derive(Parser, Debug)]
#[command(name = "turbulance")]
#[command(about = "Turbulance DSL compiler for advanced sports analysis")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Enable debug mode
    #[arg(short, long)]
    debug: bool,
    
    /// Output directory for generated files
    #[arg(short, long, default_value = "./output")]
    output: PathBuf,
    
    /// Compilation target
    #[arg(short, long, default_value = "python")]
    target: String,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compile Turbulance source files
    Compile {
        /// Input source files
        files: Vec<PathBuf>,
        
        /// Watch mode - recompile on file changes
        #[arg(short, long)]
        watch: bool,
        
        /// Optimize generated code
        #[arg(short = 'O', long)]
        optimize: bool,
    },
    
    /// Validate Turbulance source files
    Validate {
        /// Input source files
        files: Vec<PathBuf>,
        
        /// Show detailed validation output
        #[arg(short, long)]
        detailed: bool,
    },
    
    /// Format Turbulance source code
    Format {
        /// Input source files
        files: Vec<PathBuf>,
        
        /// Format in place
        #[arg(short, long)]
        in_place: bool,
    },
    
    /// Analyze code complexity and structure
    Analyze {
        /// Input source files
        files: Vec<PathBuf>,
        
        /// Generate analysis report
        #[arg(short, long)]
        report: bool,
    },
    
    /// Run interactive REPL
    Repl {
        /// Load initial script
        #[arg(short, long)]
        script: Option<PathBuf>,
    },
    
    /// Generate documentation
    Doc {
        /// Input source files
        files: Vec<PathBuf>,
        
        /// Output format (html, markdown, json)
        #[arg(short, long, default_value = "html")]
        format: String,
    },
    
    /// Show language features and examples
    Features,
    
    /// Create a new Turbulance project
    New {
        /// Project name
        name: String,
        
        /// Project template (basic, sports, research)
        #[arg(short, long, default_value = "basic")]
        template: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    env_logger::Builder::from_env(Env::default().default_filter_or(
        if cli.verbose { "debug" } else { "info" }
    )).init();
    
    info!("Turbulance Compiler v{}", env!("CARGO_PKG_VERSION"));
    
    // Parse target
    let target = match cli.target.to_lowercase().as_str() {
        "python" | "py" => CompilationTarget::Python,
        "rust" | "rs" => CompilationTarget::Rust,
        "javascript" | "js" => CompilationTarget::JavaScript,
        _ => return Err(anyhow!("Unsupported target: {}", cli.target)),
    };
    
    // Create compiler
    let compiler = TurbulanceCompiler::new(cli.output.to_string_lossy().to_string())
        .with_debug(cli.debug)
        .with_target(target);
    
    match cli.command {
        Commands::Compile { files, watch, optimize } => {
            compile_command(&compiler, files, watch, optimize)
        }
        Commands::Validate { files, detailed } => {
            validate_command(&compiler, files, detailed)
        }
        Commands::Format { files, in_place } => {
            format_command(files, in_place)
        }
        Commands::Analyze { files, report } => {
            analyze_command(files, report)
        }
        Commands::Repl { script } => {
            repl_command(script)
        }
        Commands::Doc { files, format } => {
            doc_command(files, format)
        }
        Commands::Features => {
            features_command()
        }
        Commands::New { name, template } => {
            new_command(name, template)
        }
    }
}

fn compile_command(
    compiler: &TurbulanceCompiler, 
    files: Vec<PathBuf>, 
    watch: bool, 
    _optimize: bool
) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow!("No input files specified"));
    }
    
    if watch {
        info!("Watch mode not yet implemented");
        return Err(anyhow!("Watch mode not yet implemented"));
    }
    
    for file in files {
        info!("Compiling: {}", file.display());
        
        match compiler.compile_file(&file) {
            Ok(result) => {
                info!("✓ Compiled successfully");
                info!("  Output: {}", result.output_path);
                info!("  Parse time: {:.2}ms", result.metrics.parse_time_ms);
                info!("  Code generation time: {:.2}ms", result.metrics.codegen_time_ms);
                info!("  Total time: {:.2}ms", result.metrics.total_time_ms);
                info!("  Output size: {} bytes", result.metrics.output_size_bytes);
                
                if !result.warnings.is_empty() {
                    warn!("Warnings:");
                    for warning in result.warnings {
                        warn!("  {}", warning);
                    }
                }
            }
            Err(e) => {
                error!("✗ Compilation failed: {}", e);
                return Err(e);
            }
        }
    }
    
    info!("All files compiled successfully");
    Ok(())
}

fn validate_command(compiler: &TurbulanceCompiler, files: Vec<PathBuf>, detailed: bool) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow!("No input files specified"));
    }
    
    let mut all_valid = true;
    
    for file in files {
        info!("Validating: {}", file.display());
        
        let source = std::fs::read_to_string(&file)?;
        match compiler.validate_source(&source) {
            Ok(result) => {
                if result.is_valid {
                    info!("✓ Valid");
                } else {
                    error!("✗ Invalid");
                    all_valid = false;
                    
                    if !result.syntax_errors.is_empty() {
                        error!("Syntax errors:");
                        for error in result.syntax_errors {
                            error!("  {}", error);
                        }
                    }
                    
                    if !result.semantic_errors.is_empty() {
                        error!("Semantic errors:");
                        for error in result.semantic_errors {
                            error!("  {}", error);
                        }
                    }
                }
                
                if detailed && !result.warnings.is_empty() {
                    warn!("Warnings:");
                    for warning in result.warnings {
                        warn!("  {}", warning);
                    }
                }
            }
            Err(e) => {
                error!("✗ Validation error: {}", e);
                all_valid = false;
            }
        }
    }
    
    if all_valid {
        info!("All files are valid");
        Ok(())
    } else {
        Err(anyhow!("Some files have validation errors"))
    }
}

fn format_command(files: Vec<PathBuf>, in_place: bool) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow!("No input files specified"));
    }
    
    for file in files {
        info!("Formatting: {}", file.display());
        
        let source = std::fs::read_to_string(&file)?;
        let formatted = utils::format_source(&source)?;
        
        if in_place {
            std::fs::write(&file, formatted)?;
            info!("✓ Formatted in place");
        } else {
            println!("{}", formatted);
        }
    }
    
    Ok(())
}

fn analyze_command(files: Vec<PathBuf>, report: bool) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow!("No input files specified"));
    }
    
    for file in files {
        info!("Analyzing: {}", file.display());
        
        let source = std::fs::read_to_string(&file)?;
        
        // Extract propositions
        let propositions = utils::find_propositions(&source)?;
        info!("Propositions found: {}", propositions.len());
        for prop in propositions {
            info!("  {} (motions: {}, contexts: {})", prop.name, prop.motion_count, prop.context_count);
        }
        
        // Analyze complexity
        let complexity = utils::analyze_complexity(&source)?;
        info!("Complexity metrics:");
        info!("  Statements: {}", complexity.statement_count);
        info!("  Functions: {}", complexity.function_count);
        info!("  Propositions: {}", complexity.proposition_count);
        info!("  Evidence blocks: {}", complexity.evidence_count);
        info!("  Cyclomatic complexity: {}", complexity.cyclomatic_complexity);
        
        if report {
            // Generate detailed report
            let report_content = generate_analysis_report(&file, &complexity, &source)?;
            let report_path = file.with_extension("analysis.md");
            std::fs::write(&report_path, report_content)?;
            info!("✓ Analysis report written to: {}", report_path.display());
        }
    }
    
    Ok(())
}

fn repl_command(_script: Option<PathBuf>) -> Result<()> {
    println!("Turbulance REPL v{}", env!("CARGO_PKG_VERSION"));
    println!("Type 'help' for commands, 'exit' to quit");
    
    // TODO: Implement REPL
    warn!("REPL not yet implemented");
    Ok(())
}

fn doc_command(files: Vec<PathBuf>, format: String) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow!("No input files specified"));
    }
    
    for file in files {
        info!("Generating documentation for: {}", file.display());
        
        let source = std::fs::read_to_string(&file)?;
        let docs = utils::extract_documentation(&source)?;
        
        match format.as_str() {
            "markdown" | "md" => {
                let doc_content = generate_markdown_docs(&docs, &file)?;
                let doc_path = file.with_extension("md");
                std::fs::write(&doc_path, doc_content)?;
                info!("✓ Markdown documentation written to: {}", doc_path.display());
            }
            "html" => {
                let doc_content = generate_html_docs(&docs, &file)?;
                let doc_path = file.with_extension("html");
                std::fs::write(&doc_path, doc_content)?;
                info!("✓ HTML documentation written to: {}", doc_path.display());
            }
            "json" => {
                let doc_content = generate_json_docs(&docs)?;
                let doc_path = file.with_extension("doc.json");
                std::fs::write(&doc_path, doc_content)?;
                info!("✓ JSON documentation written to: {}", doc_path.display());
            }
            _ => return Err(anyhow!("Unsupported documentation format: {}", format)),
        }
    }
    
    Ok(())
}

fn features_command() -> Result<()> {
    println!("Turbulance Language Features:");
    println!();
    
    let features = TurbulanceCompiler::supported_features();
    for feature in features {
        println!("  ✓ {}", feature);
    }
    
    println!();
    println!("Example usage:");
    println!();
    println!(r#"
proposition SprintAnalysis:
    context athlete_data = load_athlete_profile("sprinter.json")
    
    motion OptimalTechnique("Athlete demonstrates optimal sprint technique"):
        target_stride_frequency: 4.5..5.2
        ground_contact_time: 0.08..0.12
    
    within video_analysis_results:
        given stride_frequency in OptimalTechnique.target_stride_frequency:
            support OptimalTechnique with_confidence(
                measurement_quality * environmental_factor
            )

bayesian_network PerformanceNetwork:
    nodes:
        - technique: TechniqueEvidence(confidence_threshold: 0.8)
        - biomechanics: BiomechanicalEvidence(precision: 0.02)
        - performance: PerformanceEvidence(measurement_accuracy: 0.95)
    
    edges:
        - technique -> performance: causal_strength(0.85, fuzziness: 0.15)
        - biomechanics -> technique: influence_strength(0.75, fuzziness: 0.2)
"#);
    
    println!();
    println!("For more information, visit: https://github.com/your-repo/turbulance");
    
    Ok(())
}

fn new_command(name: String, template: String) -> Result<()> {
    info!("Creating new Turbulance project: {}", name);
    
    let project_dir = PathBuf::from(&name);
    if project_dir.exists() {
        return Err(anyhow!("Directory '{}' already exists", name));
    }
    
    std::fs::create_dir_all(&project_dir)?;
    
    // Create project structure
    std::fs::create_dir_all(project_dir.join("src"))?;
    std::fs::create_dir_all(project_dir.join("data"))?;
    std::fs::create_dir_all(project_dir.join("output"))?;
    
    let template_content = match template.as_str() {
        "basic" => generate_basic_template(&name),
        "sports" => generate_sports_template(&name),
        "research" => generate_research_template(&name),
        _ => return Err(anyhow!("Unknown template: {}", template)),
    };
    
    let main_file = project_dir.join("src").join("main.tbn");
    std::fs::write(main_file, template_content)?;
    
    // Create README
    let readme_content = format!(r#"# {}

A Turbulance project for sports analysis and evidence-based reasoning.

## Structure

- `src/` - Turbulance source files
- `data/` - Input data files (videos, sensor data, etc.)
- `output/` - Generated analysis results

## Usage

```bash
# Compile the project
turbulance compile src/main.tbn

# Validate the code
turbulance validate src/main.tbn

# Run analysis
python output/main.py
```

## Template: {}

This project was created from the '{}' template.
"#, name, template, template);
    
    std::fs::write(project_dir.join("README.md"), readme_content)?;
    
    info!("✓ Project '{}' created successfully", name);
    info!("  Template: {}", template);
    info!("  Directory: {}", project_dir.display());
    
    Ok(())
}

fn generate_analysis_report(
    file: &PathBuf, 
    complexity: &turbulance::ComplexityMetrics, 
    source: &str
) -> Result<String> {
    let mut report = String::new();
    
    report.push_str(&format!("# Analysis Report: {}\n\n", file.display()));
    report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    
    report.push_str("## Complexity Metrics\n\n");
    report.push_str(&format!("- **Statements**: {}\n", complexity.statement_count));
    report.push_str(&format!("- **Functions**: {}\n", complexity.function_count));
    report.push_str(&format!("- **Propositions**: {}\n", complexity.proposition_count));
    report.push_str(&format!("- **Evidence blocks**: {}\n", complexity.evidence_count));
    report.push_str(&format!("- **Cyclomatic complexity**: {}\n", complexity.cyclomatic_complexity));
    
    report.push_str("\n## Source Statistics\n\n");
    report.push_str(&format!("- **Lines of code**: {}\n", source.lines().count()));
    report.push_str(&format!("- **File size**: {} bytes\n", source.len()));
    
    Ok(report)
}

fn generate_markdown_docs(docs: &[turbulance::DocumentationBlock], file: &PathBuf) -> Result<String> {
    let mut content = String::new();
    
    content.push_str(&format!("# Documentation: {}\n\n", file.display()));
    
    for doc in docs {
        content.push_str(&format!("## Line {}\n\n", doc.line_number));
        content.push_str(&format!("{}\n\n", doc.content));
    }
    
    Ok(content)
}

fn generate_html_docs(docs: &[turbulance::DocumentationBlock], file: &PathBuf) -> Result<String> {
    let mut content = String::new();
    
    content.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    content.push_str(&format!("<title>Documentation: {}</title>\n", file.display()));
    content.push_str("</head>\n<body>\n");
    content.push_str(&format!("<h1>Documentation: {}</h1>\n", file.display()));
    
    for doc in docs {
        content.push_str(&format!("<h2>Line {}</h2>\n", doc.line_number));
        content.push_str(&format!("<p>{}</p>\n", doc.content));
    }
    
    content.push_str("</body>\n</html>");
    
    Ok(content)
}

fn generate_json_docs(docs: &[turbulance::DocumentationBlock]) -> Result<String> {
    serde_json::to_string_pretty(docs).map_err(|e| anyhow!("JSON serialization error: {}", e))
}

fn generate_basic_template(name: &str) -> String {
    format!(r#"// {} - Basic Turbulance Project
// This is a basic template for getting started with Turbulance

/// Basic data analysis example
funxn analyze_data(data_path: String) -> AnalysisResult:
    item dataset = load_data(data_path)
    item results = process_dataset(dataset)
    return results

/// Simple proposition for demonstration
proposition BasicAnalysis:
    context input_data = load_data("data/sample.csv")
    
    motion DataQuality("Input data meets quality standards"):
        completeness_threshold: 0.95
        accuracy_threshold: 0.9
    
    within input_data:
        given data_completeness() > DataQuality.completeness_threshold:
            support DataQuality with_confidence(data_quality_score())

// Main execution
item main_analysis = BasicAnalysis
item results = analyze_data("data/input.csv")
"#, name)
}

fn generate_sports_template(name: &str) -> String {
    format!(r#"// {} - Sports Analysis Project
// Turbulance template for sports video analysis

/// Sprint performance analysis
proposition SprintPerformance:
    context athlete_profile = load_athlete_profile("data/athlete.json")
    context video_data = load_video("data/sprint_video.mp4")
    
    motion OptimalTechnique("Athlete demonstrates optimal sprint technique"):
        stride_frequency_range: 4.5..5.2
        ground_contact_time_range: 0.08..0.12
        stride_length_optimal: true
    
    motion EfficientBiomechanics("Biomechanical efficiency is high"):
        energy_efficiency_threshold: 0.85
        joint_coordination_score: 0.9
    
    within video_analysis_results:
        biomechanical sprint_kinematics:
            stride_analysis = analyze_stride_patterns(video_data)
            
            given stride_analysis.frequency in OptimalTechnique.stride_frequency_range:
                given stride_analysis.ground_contact_time in OptimalTechnique.ground_contact_time_range:
                    support OptimalTechnique with_confidence(
                        stride_analysis.consistency_score * 0.8 +
                        measurement_confidence * 0.2
                    )

/// Bayesian network for performance prediction
bayesian_network PerformanceNetwork:
    nodes:
        - technique: TechniqueEvidence(confidence_threshold: 0.8)
        - biomechanics: BiomechanicalEvidence(precision: 0.02)
        - performance: PerformanceEvidence(measurement_accuracy: 0.95)
    
    edges:
        - technique -> performance: causal_strength(0.85, fuzziness: 0.15)
        - biomechanics -> technique: influence_strength(0.75, fuzziness: 0.2)

// Initialize analysis pipeline
item performance_analysis = SprintPerformance
item network = PerformanceNetwork
"#, name)
}

fn generate_research_template(name: &str) -> String {
    format!(r#"// {} - Research Analysis Project
// Advanced Turbulance template for research applications

/// Research hypothesis testing framework
proposition ResearchHypothesis:
    context experimental_data = load_experimental_data("data/experiment.csv")
    context control_data = load_control_data("data/control.csv")
    
    motion SignificantEffect("Experimental treatment shows significant effect"):
        alpha_level: 0.05
        effect_size_threshold: 0.5
        power_threshold: 0.8
    
    motion ValidityMaintained("Experimental validity is maintained"):
        internal_validity_score: 0.9
        external_validity_score: 0.8
        construct_validity_score: 0.85
    
    within experimental_data:
        statistical_analysis effect_testing:
            test_results = perform_significance_test(experimental_data, control_data)
            effect_size = calculate_effect_size(test_results)
            
            given test_results.p_value < SignificantEffect.alpha_level:
                given effect_size > SignificantEffect.effect_size_threshold:
                    support SignificantEffect with_evidence(
                        p_value: test_results.p_value,
                        effect_size: effect_size,
                        confidence_interval: test_results.ci
                    )

/// Evidence integration system
evidence ResearchEvidence:
    sources:
        - quantitative_data: StatisticalData(n_samples: 100, power: 0.95)
        - qualitative_data: QualitativeData(coding_reliability: 0.9)
        - meta_analysis: MetaAnalysisData(k_studies: 15, heterogeneity: 0.3)
    
    collection_methodology:
        randomization: true
        blinding: double_blind
        control_group: true
        
    validation_criteria:
        statistical_assumptions: check_normality_homogeneity
        measurement_reliability: cronbach_alpha > 0.8
        construct_validity: confirmatory_factor_analysis

/// Metacognitive analysis of research quality
metacognitive ResearchQuality:
    track:
        - methodology_rigor: assess_experimental_design
        - bias_identification: systematic_bias_detection
        - reproducibility: replication_potential_assessment
    
    evaluate:
        - internal_validity: threats_to_validity_analysis
        - external_validity: generalizability_assessment
        - statistical_conclusion_validity: power_analysis
    
    adapt:
        given methodology_rigor < 0.8:
            recommend_design_improvements()
            increase_sample_size()
        given bias_detected:
            apply_bias_correction_methods()
            implement_additional_controls()

// Initialize research framework
item hypothesis_test = ResearchHypothesis
item evidence_base = ResearchEvidence
item quality_monitor = ResearchQuality
"#, name)
} 