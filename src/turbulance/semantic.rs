use crate::turbulance::ast::*;
use std::collections::HashMap;
use anyhow::{Result, anyhow};

/// Semantic analyzer for Turbulance programs
pub struct SemanticAnalyzer {
    /// Symbol table for tracking variables, functions, etc.
    symbol_table: HashMap<String, SymbolInfo>,
    /// Current scope name
    current_scope: String,
    /// Error accumulator
    errors: Vec<String>,
    /// Warning accumulator
    warnings: Vec<String>,
    /// Type inference engine
    type_inference: TypeInferenceEngine,
}

/// Type inference engine
pub struct TypeInferenceEngine {
    /// Known type mappings
    type_map: HashMap<String, TypeInfo>,
    /// Type constraints
    constraints: Vec<TypeConstraint>,
}

/// Information about a type
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: String,
    pub category: TypeCategory,
    pub properties: HashMap<String, TypeInfo>,
}

/// Type categories
#[derive(Debug, Clone, PartialEq)]
pub enum TypeCategory {
    Primitive,      // int, float, bool, string
    Collection,     // list, dict
    Function,       // function types
    Evidence,       // evidence types
    Proposition,    // proposition types
    BayesianNode,   // Bayesian network node types
    FuzzyValue,     // fuzzy logic types
    Unknown,        // unresolved types
}

/// Type constraint for inference
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub left: String,
    pub right: String,
    pub constraint_type: ConstraintType,
    pub location: Span,
}

/// Types of type constraints
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    Equal,          // left == right
    Subtype,        // left <: right
    FunctionCall,   // left(args) -> right
    MemberAccess,   // left.member == right
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            symbol_table: HashMap::new(),
            current_scope: "global".to_string(),
            errors: Vec::new(),
            warnings: Vec::new(),
            type_inference: TypeInferenceEngine::new(),
        }
    }

    /// Analyze a program and return semantic analysis results
    pub fn analyze(&mut self, program: &Program) -> Result<SemanticAnalysisResult> {
        // Initialize built-in types
        self.initialize_builtin_types();
        
        // First pass: collect all declarations
        for statement in &program.statements {
            self.collect_declarations(statement)?;
        }
        
        // Second pass: type checking and semantic validation
        for statement in &program.statements {
            self.analyze_statement(statement)?;
        }
        
        // Resolve type constraints
        self.type_inference.resolve_constraints()?;
        
        Ok(SemanticAnalysisResult {
            errors: self.errors.clone(),
            warnings: self.warnings.clone(),
            symbol_table: self.symbol_table.clone(),
        })
    }

    /// Initialize built-in types and functions
    fn initialize_builtin_types(&mut self) {
        let builtin_types = vec![
            ("int", TypeCategory::Primitive),
            ("float", TypeCategory::Primitive),
            ("bool", TypeCategory::Primitive),
            ("string", TypeCategory::Primitive),
            ("list", TypeCategory::Collection),
            ("dict", TypeCategory::Collection),
        ];

        for (name, category) in builtin_types {
            self.type_inference.type_map.insert(name.to_string(), TypeInfo {
                name: name.to_string(),
                category,
                properties: HashMap::new(),
            });
        }

        // Built-in functions
        let builtin_functions = vec![
            "load_data",
            "process_dataset", 
            "analyze_stride_patterns",
            "perform_significance_test",
            "calculate_effect_size",
        ];

        for func_name in builtin_functions {
            self.symbol_table.insert(func_name.to_string(), SymbolInfo {
                name: func_name.to_string(),
                symbol_type: SymbolType::Function,
                scope: "global".to_string(),
                defined_at: Position { line: 0, column: 0, file: None },
            });
        }
    }

    /// Collect all top-level declarations
    fn collect_declarations(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::VariableDeclaration(var_decl) => {
                self.add_symbol(&var_decl.name, SymbolType::Variable("unknown".to_string()), &var_decl.span.start)?;
            }
            Statement::FunctionDeclaration(func_decl) => {
                self.add_symbol(&func_decl.name, SymbolType::Function, &func_decl.span.start)?;
            }
            Statement::PropositionDeclaration(prop_decl) => {
                self.add_symbol(&prop_decl.name, SymbolType::Proposition, &prop_decl.span.start)?;
            }
            Statement::EvidenceDeclaration(evidence_decl) => {
                self.add_symbol(&evidence_decl.name, SymbolType::Evidence, &evidence_decl.span.start)?;
            }
            Statement::BayesianNetworkDeclaration(bn_decl) => {
                self.add_symbol(&bn_decl.name, SymbolType::BayesianNetwork, &bn_decl.span.start)?;
            }
            Statement::FuzzySystemDeclaration(fs_decl) => {
                self.add_symbol(&fs_decl.name, SymbolType::FuzzySystem, &fs_decl.span.start)?;
            }
            _ => {} // Skip other statements in declaration collection
        }
        Ok(())
    }

    /// Add a symbol to the symbol table
    fn add_symbol(&mut self, name: &str, symbol_type: SymbolType, position: &Position) -> Result<()> {
        if self.symbol_table.contains_key(name) {
            self.errors.push(format!("Symbol '{}' is already defined", name));
            return Ok(());
        }

        self.symbol_table.insert(name.to_string(), SymbolInfo {
            name: name.to_string(),
            symbol_type,
            scope: self.current_scope.clone(),
            defined_at: position.clone(),
        });

        Ok(())
    }

    /// Analyze a statement for semantic correctness
    fn analyze_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::VariableDeclaration(var_decl) => {
                self.analyze_variable_declaration(var_decl)
            }
            Statement::FunctionDeclaration(func_decl) => {
                self.analyze_function_declaration(func_decl)
            }
            Statement::PropositionDeclaration(prop_decl) => {
                self.analyze_proposition_declaration(prop_decl)
            }
            Statement::EvidenceDeclaration(evidence_decl) => {
                self.analyze_evidence_declaration(evidence_decl)
            }
            Statement::BayesianNetworkDeclaration(bn_decl) => {
                self.analyze_bayesian_network_declaration(bn_decl)
            }
            Statement::Assignment(assignment) => {
                self.analyze_assignment(assignment)
            }
            Statement::Expression(expr) => {
                self.analyze_expression(expr).map(|_| ())
            }
            Statement::Within(within_stmt) => {
                self.analyze_within_statement(within_stmt)
            }
            _ => {
                // For unimplemented statements, just add a warning
                self.warnings.push("Statement type not fully implemented in semantic analysis".to_string());
                Ok(())
            }
        }
    }

    /// Analyze variable declaration
    fn analyze_variable_declaration(&mut self, var_decl: &VariableDeclaration) -> Result<()> {
        // Analyze the initializer expression
        let expr_type = self.analyze_expression(&var_decl.initializer)?;
        
        // Update symbol type information
        if let Some(symbol) = self.symbol_table.get_mut(&var_decl.name) {
            symbol.symbol_type = SymbolType::Variable(expr_type.name);
        }

        // If there's a type annotation, check compatibility
        if let Some(type_annotation) = &var_decl.type_annotation {
            let expected_type = self.resolve_type_expression(type_annotation)?;
            if !self.are_types_compatible(&expr_type, &expected_type) {
                self.errors.push(format!(
                    "Type mismatch in variable '{}': expected {}, got {}",
                    var_decl.name, expected_type.name, expr_type.name
                ));
            }
        }

        Ok(())
    }

    /// Analyze function declaration
    fn analyze_function_declaration(&mut self, func_decl: &FunctionDeclaration) -> Result<()> {
        // Create new scope for function
        let old_scope = self.current_scope.clone();
        self.current_scope = format!("{}::{}", old_scope, func_decl.name);

        // Add parameters to symbol table
        for param in &func_decl.parameters {
            let param_type = if let Some(type_ann) = &param.type_annotation {
                self.resolve_type_expression(type_ann)?.name
            } else {
                "unknown".to_string()
            };
            
            self.add_symbol(&param.name, SymbolType::Variable(param_type), &param.span.start)?;
        }

        // Analyze function body
        self.analyze_block_statement(&func_decl.body)?;

        // Restore previous scope
        self.current_scope = old_scope;

        Ok(())
    }

    /// Analyze proposition declaration
    fn analyze_proposition_declaration(&mut self, prop_decl: &PropositionDeclaration) -> Result<()> {
        // Validate contexts
        for context in &prop_decl.contexts {
            self.analyze_expression(&context.value)?;
        }

        // Validate motions
        for motion in &prop_decl.motions {
            // Check motion properties
            for (prop_name, prop_value) in &motion.properties {
                let prop_type = self.analyze_expression(prop_value)?;
                
                // Validate property types based on common motion properties
                match prop_name.as_str() {
                    "threshold" | "confidence" | "weight" => {
                        if prop_type.category != TypeCategory::Primitive {
                            self.warnings.push(format!(
                                "Motion property '{}' should be numeric, got {}",
                                prop_name, prop_type.name
                            ));
                        }
                    }
                    _ => {} // Allow other properties
                }
            }
        }

        // Analyze within blocks
        for within_block in &prop_decl.within_blocks {
            self.analyze_within_statement(within_block)?;
        }

        Ok(())
    }

    /// Analyze evidence declaration
    fn analyze_evidence_declaration(&mut self, evidence_decl: &EvidenceDeclaration) -> Result<()> {
        // Validate evidence sections
        for (section_name, section) in &evidence_decl.sections {
            match section {
                EvidenceSection::Simple(expr) => {
                    self.analyze_expression(expr)?;
                }
                EvidenceSection::Complex(properties) => {
                    for (prop_name, prop_value) in properties {
                        self.analyze_expression(prop_value)?;
                        
                        // Validate common evidence properties
                        match prop_name.as_str() {
                            "confidence" | "reliability" | "accuracy" => {
                                // Should be between 0 and 1
                                if let Expression::NumberLiteral(val, _) = prop_value {
                                    if *val < 0.0 || *val > 1.0 {
                                        self.warnings.push(format!(
                                            "Evidence property '{}' should be between 0 and 1, got {}",
                                            prop_name, val
                                        ));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Analyze Bayesian network declaration
    fn analyze_bayesian_network_declaration(&mut self, bn_decl: &BayesianNetworkDeclaration) -> Result<()> {
        // Check that all edge references point to valid nodes
        let node_names: std::collections::HashSet<_> = bn_decl.nodes.iter().map(|n| &n.name).collect();
        
        for edge in &bn_decl.edges {
            if !node_names.contains(&edge.from) {
                self.errors.push(format!(
                    "Bayesian network edge references undefined node: '{}'",
                    edge.from
                ));
            }
            if !node_names.contains(&edge.to) {
                self.errors.push(format!(
                    "Bayesian network edge references undefined node: '{}'",
                    edge.to
                ));
            }
            
            // Validate edge strength
            let strength_type = self.analyze_expression(&edge.strength)?;
            if strength_type.category != TypeCategory::Primitive {
                self.warnings.push("Edge strength should be numeric".to_string());
            }
        }

        Ok(())
    }

    /// Analyze assignment
    fn analyze_assignment(&mut self, assignment: &Assignment) -> Result<()> {
        // Check if target variable exists or needs to be created
        if !self.symbol_table.contains_key(&assignment.target) {
            self.warnings.push(format!(
                "Assignment to undeclared variable: '{}'",
                assignment.target
            ));
        }

        // Analyze the value expression
        self.analyze_expression(&assignment.value)?;

        Ok(())
    }

    /// Analyze within statement
    fn analyze_within_statement(&mut self, within_stmt: &WithinStatement) -> Result<()> {
        // Analyze target expression
        let target_type = self.analyze_expression(&within_stmt.target)?;
        
        // Validate that target is iterable
        if target_type.category != TypeCategory::Collection && target_type.name != "unknown" {
            self.warnings.push(format!(
                "Within statement target should be iterable, got {}",
                target_type.name
            ));
        }

        // Analyze within body contents
        for content in &within_stmt.body.contents {
            match content {
                WithinContent::Statement(stmt) => {
                    self.analyze_statement(stmt)?;
                }
                WithinContent::PatternMatching(pattern_block) => {
                    self.analyze_expression(&pattern_block.condition)?;
                    self.analyze_block_statement(&pattern_block.body)?;
                }
                _ => {
                    // Placeholder for other within content types
                }
            }
        }

        Ok(())
    }

    /// Analyze block statement
    fn analyze_block_statement(&mut self, block: &BlockStatement) -> Result<()> {
        for statement in &block.statements {
            self.analyze_statement(statement)?;
        }
        Ok(())
    }

    /// Analyze an expression and return its type
    fn analyze_expression(&mut self, expr: &Expression) -> Result<TypeInfo> {
        match expr {
            Expression::Identifier(name, _) => {
                if let Some(symbol) = self.symbol_table.get(name) {
                    match &symbol.symbol_type {
                        SymbolType::Variable(type_name) => {
                            Ok(self.get_type_info(type_name))
                        }
                        _ => Ok(self.get_type_info("unknown"))
                    }
                } else {
                    self.warnings.push(format!("Reference to undefined variable: '{}'", name));
                    Ok(self.get_type_info("unknown"))
                }
            }
            Expression::StringLiteral(_, _) => Ok(self.get_type_info("string")),
            Expression::NumberLiteral(_, _) => Ok(self.get_type_info("float")),
            Expression::BooleanLiteral(_, _) => Ok(self.get_type_info("bool")),
            Expression::List(items, _) => {
                // Analyze all items and try to infer element type
                let mut element_types = Vec::new();
                for item in items {
                    element_types.push(self.analyze_expression(item)?);
                }
                
                // For now, just return generic list type
                Ok(self.get_type_info("list"))
            }
            Expression::Dictionary(_, _) => Ok(self.get_type_info("dict")),
            Expression::FunctionCall(call) => {
                // Analyze arguments
                for arg in &call.arguments {
                    self.analyze_expression(arg)?;
                }
                
                // For now, return unknown type
                // TODO: Implement proper function signature lookup
                Ok(self.get_type_info("unknown"))
            }
            Expression::MemberAccess(access) => {
                let _object_type = self.analyze_expression(&access.object)?;
                // TODO: Implement proper member type resolution
                Ok(self.get_type_info("unknown"))
            }
            Expression::BinaryOp(op) => {
                let left_type = self.analyze_expression(&op.left)?;
                let right_type = self.analyze_expression(&op.right)?;
                
                // Simple type inference for binary operations
                match op.operator {
                    BinaryOperator::Add | BinaryOperator::Sub | 
                    BinaryOperator::Mul | BinaryOperator::Div => {
                        if left_type.category == TypeCategory::Primitive && right_type.category == TypeCategory::Primitive {
                            Ok(self.get_type_info("float"))
                        } else {
                            Ok(self.get_type_info("unknown"))
                        }
                    }
                    BinaryOperator::Eq | BinaryOperator::Ne | 
                    BinaryOperator::Lt | BinaryOperator::Gt |
                    BinaryOperator::Le | BinaryOperator::Ge |
                    BinaryOperator::And | BinaryOperator::Or => {
                        Ok(self.get_type_info("bool"))
                    }
                    _ => Ok(self.get_type_info("unknown"))
                }
            }
            Expression::UnaryOp(op) => {
                let operand_type = self.analyze_expression(&op.operand)?;
                match op.operator {
                    UnaryOperator::Not => Ok(self.get_type_info("bool")),
                    UnaryOperator::Minus => Ok(operand_type),
                }
            }
            Expression::Range(_, _, _) => Ok(self.get_type_info("range")),
        }
    }

    /// Resolve a type expression to type info
    fn resolve_type_expression(&self, type_expr: &TypeExpression) -> Result<TypeInfo> {
        match type_expr {
            TypeExpression::Named { name, .. } => {
                Ok(self.get_type_info(name))
            }
            TypeExpression::Optional(inner) => {
                let inner_type = self.resolve_type_expression(inner)?;
                Ok(TypeInfo {
                    name: format!("Optional<{}>", inner_type.name),
                    category: inner_type.category,
                    properties: inner_type.properties,
                })
            }
            TypeExpression::Function { .. } => {
                Ok(TypeInfo {
                    name: "function".to_string(),
                    category: TypeCategory::Function,
                    properties: HashMap::new(),
                })
            }
        }
    }

    /// Get type info for a type name
    fn get_type_info(&self, type_name: &str) -> TypeInfo {
        self.type_inference.type_map.get(type_name)
            .cloned()
            .unwrap_or_else(|| TypeInfo {
                name: type_name.to_string(),
                category: TypeCategory::Unknown,
                properties: HashMap::new(),
            })
    }

    /// Check if two types are compatible
    fn are_types_compatible(&self, actual: &TypeInfo, expected: &TypeInfo) -> bool {
        if actual.name == expected.name {
            return true;
        }
        
        // Allow unknown types to be compatible with anything
        if actual.category == TypeCategory::Unknown || expected.category == TypeCategory::Unknown {
            return true;
        }
        
        // Add more sophisticated type compatibility rules here
        false
    }
}

impl TypeInferenceEngine {
    pub fn new() -> Self {
        Self {
            type_map: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Resolve all type constraints
    pub fn resolve_constraints(&mut self) -> Result<()> {
        // Simple constraint resolution
        // TODO: Implement proper constraint solving algorithm
        
        for constraint in &self.constraints {
            match constraint.constraint_type {
                ConstraintType::Equal => {
                    // Try to unify types
                    self.unify_types(&constraint.left, &constraint.right)?;
                }
                _ => {
                    // Other constraint types not yet implemented
                }
            }
        }
        
        Ok(())
    }

    /// Unify two types
    fn unify_types(&mut self, left: &str, right: &str) -> Result<()> {
        // Simple unification - just update unknown types
        if left == "unknown" || right == "unknown" {
            return Ok(());
        }
        
        if left != right {
            return Err(anyhow!("Cannot unify types {} and {}", left, right));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_analyzer_creation() {
        let analyzer = SemanticAnalyzer::new();
        assert!(!analyzer.symbol_table.is_empty()); // Should have built-ins
    }

    #[test]
    fn test_variable_declaration_analysis() {
        let mut analyzer = SemanticAnalyzer::new();
        
        let var_decl = VariableDeclaration {
            name: "test_var".to_string(),
            type_annotation: None,
            initializer: Expression::NumberLiteral(42.0, Span {
                start: Position { line: 1, column: 1, file: None },
                end: Position { line: 1, column: 5, file: None },
            }),
            span: Span {
                start: Position { line: 1, column: 1, file: None },
                end: Position { line: 1, column: 20, file: None },
            },
        };
        
        let result = analyzer.analyze_variable_declaration(&var_decl);
        assert!(result.is_ok());
    }
} 