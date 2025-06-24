pub mod ast;
pub mod parser;
pub mod codegen;
pub mod semantic;
pub mod runtime;

use anyhow::{Result, anyhow};
use std::path::Path;
use std::fs;

pub use ast::*;
pub use parser::TurbulanceParser;
pub use codegen::PythonCodeGenerator;

/// Main Turbulance compiler
pub struct TurbulanceCompiler {
    /// Output directory for generated code
    output_dir: String,
    /// Enable debug output
    debug: bool,
    /// Target platform (python, rust, etc.)
    target: CompilationTarget,
}

/// Compilation targets
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationTarget {
    Python,
    Rust,
    JavaScript,
}

/// Compilation result
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Generated source code
    pub source_code: String,
    /// Output file path
    pub output_path: String,
    /// Compilation warnings
    pub warnings: Vec<String>,
    /// Performance metrics
    pub metrics: CompilationMetrics,
}

/// Compilation performance metrics
#[derive(Debug, Clone)]
pub struct CompilationMetrics {
    /// Parse time in milliseconds
    pub parse_time_ms: f64,
    /// Code generation time in milliseconds
    pub codegen_time_ms: f64,
    /// Total compilation time in milliseconds
    pub total_time_ms: f64,
    /// Number of AST nodes
    pub ast_node_count: usize,
    /// Generated code size in bytes
    pub output_size_bytes: usize,
}

impl TurbulanceCompiler {
    /// Create a new compiler instance
    pub fn new(output_dir: String) -> Self {
        Self {
            output_dir,
            debug: false,
            target: CompilationTarget::Python,
        }
    }

    /// Set debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Set compilation target
    pub fn with_target(mut self, target: CompilationTarget) -> Self {
        self.target = target;
        self
    }

    /// Compile a Turbulance source file
    pub fn compile_file<P: AsRef<Path>>(&self, source_path: P) -> Result<CompilationResult> {
        let start_time = std::time::Instant::now();
        
        // Read source file
        let source_content = fs::read_to_string(&source_path)
            .map_err(|e| anyhow!("Failed to read source file: {}", e))?;
        
        let source_path = source_path.as_ref();
        let file_name = source_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("turbulance_output");
        
        self.compile_string(&source_content, file_name, start_time)
    }

    /// Compile Turbulance source code from string
    pub fn compile_string(&self, source: &str, file_name: &str, start_time: std::time::Instant) -> Result<CompilationResult> {
        let mut warnings = Vec::new();
        
        // Parse the source code
        let parse_start = std::time::Instant::now();
        let ast = TurbulanceParser::parse_program(source)
            .map_err(|e| anyhow!("Parse error: {}", e))?;
        let parse_time = parse_start.elapsed().as_secs_f64() * 1000.0;
        
        if self.debug {
            println!("AST: {:#?}", ast);
        }

        // Semantic analysis
        let semantic_result = self.perform_semantic_analysis(&ast)?;
        warnings.extend(semantic_result.warnings);

        // Code generation
        let codegen_start = std::time::Instant::now();
        let generated_code = match self.target {
            CompilationTarget::Python => self.generate_python_code(&ast)?,
            CompilationTarget::Rust => self.generate_rust_code(&ast)?,
            CompilationTarget::JavaScript => self.generate_javascript_code(&ast)?,
        };
        let codegen_time = codegen_start.elapsed().as_secs_f64() * 1000.0;

        // Write output file
        let output_extension = match self.target {
            CompilationTarget::Python => "py",
            CompilationTarget::Rust => "rs",
            CompilationTarget::JavaScript => "js",
        };
        
        let output_path = format!("{}/{}.{}", self.output_dir, file_name, output_extension);
        
        // Ensure output directory exists
        if let Some(parent) = Path::new(&output_path).parent() {
            fs::create_dir_all(parent)?;
        }
        
        fs::write(&output_path, &generated_code)
            .map_err(|e| anyhow!("Failed to write output file: {}", e))?;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let ast_node_count = self.count_ast_nodes(&ast);

        Ok(CompilationResult {
            source_code: generated_code.clone(),
            output_path,
            warnings,
            metrics: CompilationMetrics {
                parse_time_ms: parse_time,
                codegen_time_ms: codegen_time,
                total_time_ms: total_time,
                ast_node_count,
                output_size_bytes: generated_code.len(),
            },
        })
    }

    /// Perform semantic analysis on the AST
    fn perform_semantic_analysis(&self, ast: &Program) -> Result<SemanticAnalysisResult> {
        // For now, just return empty result
        // TODO: Implement proper semantic analysis
        Ok(SemanticAnalysisResult {
            errors: Vec::new(),
            warnings: Vec::new(),
            symbol_table: std::collections::HashMap::new(),
        })
    }

    /// Generate Python code from AST
    fn generate_python_code(&self, ast: &Program) -> Result<String> {
        let mut codegen = PythonCodeGenerator::new();
        codegen.generate(ast)
    }

    /// Generate Rust code from AST (placeholder)
    fn generate_rust_code(&self, _ast: &Program) -> Result<String> {
        Ok("// Rust code generation not yet implemented\nfn main() {\n    println!(\"Hello from Turbulance!\");\n}".to_string())
    }

    /// Generate JavaScript code from AST (placeholder)
    fn generate_javascript_code(&self, _ast: &Program) -> Result<String> {
        Ok("// JavaScript code generation not yet implemented\nconsole.log('Hello from Turbulance!');".to_string())
    }

    /// Count AST nodes for metrics
    fn count_ast_nodes(&self, ast: &Program) -> usize {
        ast.statements.len() // Simplified count
    }

    /// Validate Turbulance source code
    pub fn validate_source(&self, source: &str) -> Result<ValidationResult> {
        // Parse to check for syntax errors
        let parse_result = TurbulanceParser::parse_program(source);
        
        match parse_result {
            Ok(ast) => {
                // Perform semantic analysis
                let semantic_result = self.perform_semantic_analysis(&ast)?;
                
                Ok(ValidationResult {
                    is_valid: semantic_result.errors.is_empty(),
                    syntax_errors: Vec::new(),
                    semantic_errors: semantic_result.errors,
                    warnings: semantic_result.warnings,
                })
            }
            Err(e) => {
                Ok(ValidationResult {
                    is_valid: false,
                    syntax_errors: vec![e.to_string()],
                    semantic_errors: Vec::new(),
                    warnings: Vec::new(),
                })
            }
        }
    }

    /// Get compiler version
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Get supported language features
    pub fn supported_features() -> Vec<&'static str> {
        vec![
            "propositions",
            "motions", 
            "evidence",
            "bayesian_networks",
            "fuzzy_systems",
            "sensor_fusion",
            "real_time_analysis",
            "pattern_matching",
            "causal_inference",
            "temporal_analysis",
            "optimization",
            "metacognitive_analysis",
        ]
    }
}

/// Semantic analysis result
#[derive(Debug, Clone)]
pub struct SemanticAnalysisResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub symbol_table: std::collections::HashMap<String, SymbolInfo>,
}

/// Symbol information for semantic analysis
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub symbol_type: SymbolType,
    pub scope: String,
    pub defined_at: Position,
}

/// Types of symbols
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolType {
    Variable(String), // type name
    Function,
    Proposition,
    Motion,
    Evidence,
    BayesianNetwork,
    FuzzySystem,
}

/// Source validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub syntax_errors: Vec<String>,
    pub semantic_errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Utility functions for working with Turbulance code
pub mod utils {
    use super::*;

    /// Format Turbulance source code
    pub fn format_source(source: &str) -> Result<String> {
        // Parse and regenerate with proper formatting
        let ast = TurbulanceParser::parse_program(source)?;
        
        // For now, just return the original source
        // TODO: Implement proper formatting
        Ok(source.to_string())
    }

    /// Extract documentation from Turbulance source
    pub fn extract_documentation(source: &str) -> Result<Vec<DocumentationBlock>> {
        let mut docs = Vec::new();
        
        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("///") {
                docs.push(DocumentationBlock {
                    content: trimmed.trim_start_matches("///").trim().to_string(),
                    line_number: line_num + 1,
                    block_type: DocBlockType::Function,
                });
            }
        }
        
        Ok(docs)
    }

    /// Find all propositions in source code
    pub fn find_propositions(source: &str) -> Result<Vec<PropositionInfo>> {
        let ast = TurbulanceParser::parse_program(source)?;
        let mut propositions = Vec::new();
        
        for statement in ast.statements {
            if let Statement::PropositionDeclaration(prop) = statement {
                propositions.push(PropositionInfo {
                    name: prop.name,
                    motion_count: prop.motions.len(),
                    context_count: prop.contexts.len(),
                    location: prop.span,
                });
            }
        }
        
        Ok(propositions)
    }

    /// Analyze code complexity
    pub fn analyze_complexity(source: &str) -> Result<ComplexityMetrics> {
        let ast = TurbulanceParser::parse_program(source)?;
        
        let mut metrics = ComplexityMetrics {
            statement_count: ast.statements.len(),
            proposition_count: 0,
            evidence_count: 0,
            function_count: 0,
            cyclomatic_complexity: 1, // Start with 1
            nesting_depth: 0,
        };
        
        for statement in &ast.statements {
            match statement {
                Statement::PropositionDeclaration(_) => metrics.proposition_count += 1,
                Statement::EvidenceDeclaration(_) => metrics.evidence_count += 1,
                Statement::FunctionDeclaration(_) => metrics.function_count += 1,
                Statement::If(_) | Statement::While(_) | Statement::For(_) => {
                    metrics.cyclomatic_complexity += 1;
                }
                _ => {}
            }
        }
        
        Ok(metrics)
    }
}

/// Documentation block
#[derive(Debug, Clone)]
pub struct DocumentationBlock {
    pub content: String,
    pub line_number: usize,
    pub block_type: DocBlockType,
}

/// Types of documentation blocks
#[derive(Debug, Clone, PartialEq)]
pub enum DocBlockType {
    Function,
    Proposition,
    Motion,
    Evidence,
    General,
}

/// Proposition information
#[derive(Debug, Clone)]
pub struct PropositionInfo {
    pub name: String,
    pub motion_count: usize,
    pub context_count: usize,
    pub location: Span,
}

/// Code complexity metrics
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub statement_count: usize,
    pub proposition_count: usize,
    pub evidence_count: usize,
    pub function_count: usize,
    pub cyclomatic_complexity: usize,
    pub nesting_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compiler_creation() {
        let compiler = TurbulanceCompiler::new("/tmp".to_string());
        assert_eq!(compiler.target, CompilationTarget::Python);
    }

    #[test]
    fn test_compile_simple_program() {
        let temp_dir = TempDir::new().unwrap();
        let compiler = TurbulanceCompiler::new(temp_dir.path().to_string_lossy().to_string());
        
        let source = r#"item temperature = 23.5"#;
        let result = compiler.compile_string(source, "test", std::time::Instant::now());
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation() {
        let compiler = TurbulanceCompiler::new("/tmp".to_string());
        
        let valid_source = r#"item temperature = 23.5"#;
        let result = compiler.validate_source(valid_source);
        assert!(result.is_ok());
        
        let invalid_source = r#"item temperature = "#;
        let result = compiler.validate_source(invalid_source);
        assert!(result.is_ok()); // Should return validation result, not error
    }

    #[test]
    fn test_utility_functions() {
        let source = r#"
        /// This is a test function
        funxn test_function():
            item result = 42
            return result
        "#;
        
        let docs = utils::extract_documentation(source).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "This is a test function");
    }
} 