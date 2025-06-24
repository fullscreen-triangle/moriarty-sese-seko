use pest::Parser;
use pest_derive::Parser;
use crate::turbulance::ast::*;
use std::collections::HashMap;
use anyhow::{Result, anyhow};

#[derive(Parser)]
#[grammar = "turbulance/grammar.pest"]
pub struct TurbulanceParser;

/// Main parser implementation
impl TurbulanceParser {
    /// Parse a Turbulance program from source code
    pub fn parse_program(source: &str) -> Result<Program> {
        let pairs = Self::parse(Rule::program, source)
            .map_err(|e| anyhow!("Parse error: {}", e))?;
        
        let mut statements = Vec::new();
        let mut span = Span {
            start: Position { line: 1, column: 1, file: None },
            end: Position { line: 1, column: 1, file: None },
        };

        for pair in pairs {
            match pair.as_rule() {
                Rule::program => {
                    span = Self::pair_to_span(&pair);
                    for inner_pair in pair.into_inner() {
                        match inner_pair.as_rule() {
                            Rule::statement => {
                                statements.push(Self::parse_statement(inner_pair)?);
                            }
                            Rule::EOI => break,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Program { statements, span })
    }

    /// Parse a statement
    fn parse_statement(pair: pest::iterators::Pair<Rule>) -> Result<Statement> {
        let inner = pair.into_inner().next().unwrap();
        
        match inner.as_rule() {
            Rule::variable_declaration => Ok(Statement::VariableDeclaration(Self::parse_variable_declaration(inner)?)),
            Rule::function_declaration => Ok(Statement::FunctionDeclaration(Self::parse_function_declaration(inner)?)),
            Rule::proposition_declaration => Ok(Statement::PropositionDeclaration(Self::parse_proposition_declaration(inner)?)),
            Rule::motion_declaration => Ok(Statement::MotionDeclaration(Self::parse_motion_declaration(inner)?)),
            Rule::evidence_declaration => Ok(Statement::EvidenceDeclaration(Self::parse_evidence_declaration(inner)?)),
            Rule::metacognitive_declaration => Ok(Statement::MetacognitiveDeclaration(Self::parse_metacognitive_declaration(inner)?)),
            Rule::temporal_declaration => Ok(Statement::TemporalDeclaration(Self::parse_temporal_declaration(inner)?)),
            Rule::bayesian_network_declaration => Ok(Statement::BayesianNetworkDeclaration(Self::parse_bayesian_network_declaration(inner)?)),
            Rule::sensor_fusion_declaration => Ok(Statement::SensorFusionDeclaration(Self::parse_sensor_fusion_declaration(inner)?)),
            Rule::fuzzy_system_declaration => Ok(Statement::FuzzySystemDeclaration(Self::parse_fuzzy_system_declaration(inner)?)),
            Rule::bayesian_update_declaration => Ok(Statement::BayesianUpdateDeclaration(Self::parse_bayesian_update_declaration(inner)?)),
            Rule::real_time_declaration => Ok(Statement::RealTimeDeclaration(Self::parse_real_time_declaration(inner)?)),
            Rule::optimization_framework_declaration => Ok(Statement::OptimizationFrameworkDeclaration(Self::parse_optimization_framework_declaration(inner)?)),
            Rule::genetic_optimization_declaration => Ok(Statement::GeneticOptimizationDeclaration(Self::parse_genetic_optimization_declaration(inner)?)),
            Rule::analysis_workflow_declaration => Ok(Statement::AnalysisWorkflowDeclaration(Self::parse_analysis_workflow_declaration(inner)?)),
            Rule::validation_framework_declaration => Ok(Statement::ValidationFrameworkDeclaration(Self::parse_validation_framework_declaration(inner)?)),
            Rule::pattern_registry_declaration => Ok(Statement::PatternRegistryDeclaration(Self::parse_pattern_registry_declaration(inner)?)),
            Rule::orchestration_declaration => Ok(Statement::OrchestrationDeclaration(Self::parse_orchestration_declaration(inner)?)),
            Rule::assignment_statement => Ok(Statement::Assignment(Self::parse_assignment(inner)?)),
            Rule::expression_statement => Ok(Statement::Expression(Self::parse_expression(inner.into_inner().next().unwrap())?)),
            Rule::if_statement => Ok(Statement::If(Self::parse_if_statement(inner)?)),
            Rule::for_statement => Ok(Statement::For(Self::parse_for_statement(inner)?)),
            Rule::while_statement => Ok(Statement::While(Self::parse_while_statement(inner)?)),
            Rule::within_statement => Ok(Statement::Within(Self::parse_within_statement(inner)?)),
            Rule::try_statement => Ok(Statement::Try(Self::parse_try_statement(inner)?)),
            Rule::return_statement => Ok(Statement::Return(Self::parse_return_statement(inner)?)),
            Rule::import_statement => Ok(Statement::Import(Self::parse_import_statement(inner)?)),
            Rule::block_statement => Ok(Statement::Block(Self::parse_block_statement(inner)?)),
            Rule::support_statement => Ok(Statement::Support(Self::parse_support_statement(inner)?)),
            Rule::causal_chain_declaration => Ok(Statement::CausalChain(Self::parse_causal_chain_declaration(inner)?)),
            _ => Err(anyhow!("Unexpected statement type: {:?}", inner.as_rule())),
        }
    }

    /// Parse variable declaration
    fn parse_variable_declaration(pair: pest::iterators::Pair<Rule>) -> Result<VariableDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip 'item' keyword
        inner.next();
        
        let name = inner.next().unwrap().as_str().to_string();
        
        let mut type_annotation = None;
        let mut initializer = None;
        
        for part in inner {
            match part.as_rule() {
                Rule::type_annotation => {
                    type_annotation = Some(Self::parse_type_expression(part.into_inner().next().unwrap())?);
                }
                Rule::expression => {
                    initializer = Some(Self::parse_expression(part)?);
                }
                _ => {}
            }
        }
        
        Ok(VariableDeclaration {
            name,
            type_annotation,
            initializer: initializer.unwrap(),
            span,
        })
    }

    /// Parse function declaration
    fn parse_function_declaration(pair: pest::iterators::Pair<Rule>) -> Result<FunctionDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip 'funxn' keyword
        inner.next();
        
        let name = inner.next().unwrap().as_str().to_string();
        
        let mut parameters = Vec::new();
        let mut return_type = None;
        let mut body = None;
        
        for part in inner {
            match part.as_rule() {
                Rule::parameter_list => {
                    parameters = Self::parse_parameter_list(part)?;
                }
                Rule::type_annotation => {
                    return_type = Some(Self::parse_type_expression(part.into_inner().next().unwrap())?);
                }
                Rule::block_statement => {
                    body = Some(Self::parse_block_statement(part)?);
                }
                _ => {}
            }
        }
        
        Ok(FunctionDeclaration {
            name,
            parameters,
            return_type,
            body: body.unwrap(),
            span,
        })
    }

    /// Parse parameter list
    fn parse_parameter_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Parameter>> {
        let mut parameters = Vec::new();
        
        for param_pair in pair.into_inner() {
            if param_pair.as_rule() == Rule::parameter {
                let span = Self::pair_to_span(&param_pair);
                let mut param_inner = param_pair.into_inner();
                
                let name = param_inner.next().unwrap().as_str().to_string();
                let mut type_annotation = None;
                
                if let Some(type_part) = param_inner.next() {
                    if type_part.as_rule() == Rule::type_annotation {
                        type_annotation = Some(Self::parse_type_expression(type_part.into_inner().next().unwrap())?);
                    }
                }
                
                parameters.push(Parameter {
                    name,
                    type_annotation,
                    span,
                });
            }
        }
        
        Ok(parameters)
    }

    /// Parse type expression
    fn parse_type_expression(pair: pest::iterators::Pair<Rule>) -> Result<TypeExpression> {
        match pair.as_rule() {
            Rule::identifier => Ok(TypeExpression::Named {
                name: pair.as_str().to_string(),
                generics: Vec::new(),
            }),
            _ => Ok(TypeExpression::Named {
                name: "Any".to_string(),
                generics: Vec::new(),
            }),
        }
    }

    /// Parse proposition declaration
    fn parse_proposition_declaration(pair: pest::iterators::Pair<Rule>) -> Result<PropositionDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip 'proposition' keyword
        inner.next();
        
        let name = inner.next().unwrap().as_str().to_string();
        
        let mut contexts = Vec::new();
        let mut motions = Vec::new();
        let mut within_blocks = Vec::new();
        
        // Skip colon
        inner.next();
        
        if let Some(body) = inner.next() {
            for part in body.into_inner() {
                match part.as_rule() {
                    Rule::context_declaration => {
                        contexts.push(Self::parse_context_declaration(part)?);
                    }
                    Rule::motion_declaration => {
                        motions.push(Self::parse_motion_declaration(part)?);
                    }
                    Rule::within_block => {
                        within_blocks.push(Self::parse_within_statement(part)?);
                    }
                    _ => {}
                }
            }
        }
        
        Ok(PropositionDeclaration {
            name,
            contexts,
            motions,
            within_blocks,
            span,
        })
    }

    /// Parse context declaration
    fn parse_context_declaration(pair: pest::iterators::Pair<Rule>) -> Result<ContextDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip 'context' keyword
        inner.next();
        
        let name = inner.next().unwrap().as_str().to_string();
        
        // Skip '=' operator
        inner.next();
        
        let value = Self::parse_expression(inner.next().unwrap())?;
        
        Ok(ContextDeclaration { name, value, span })
    }

    /// Parse motion declaration
    fn parse_motion_declaration(pair: pest::iterators::Pair<Rule>) -> Result<MotionDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip 'motion' keyword
        inner.next();
        
        let name = inner.next().unwrap().as_str().to_string();
        
        // Skip opening parenthesis
        inner.next();
        
        let description = inner.next().unwrap().as_str().trim_matches('"').to_string();
        
        // Skip closing parenthesis and colon
        inner.next();
        inner.next();
        
        let mut properties = HashMap::new();
        
        if let Some(body) = inner.next() {
            for part in body.into_inner() {
                if part.as_rule() == Rule::motion_property {
                    let mut prop_inner = part.into_inner();
                    let prop_name = prop_inner.next().unwrap().as_str().to_string();
                    // Skip colon
                    prop_inner.next();
                    let prop_value = Self::parse_expression(prop_inner.next().unwrap())?;
                    properties.insert(prop_name, prop_value);
                }
            }
        }
        
        Ok(MotionDeclaration {
            name,
            description,
            properties,
            span,
        })
    }

    /// Parse evidence declaration
    fn parse_evidence_declaration(pair: pest::iterators::Pair<Rule>) -> Result<EvidenceDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip 'evidence' keyword
        inner.next();
        
        let name = inner.next().unwrap().as_str().to_string();
        
        // Skip colon
        inner.next();
        
        let mut sections = HashMap::new();
        
        if let Some(body) = inner.next() {
            for part in body.into_inner() {
                if part.as_rule() == Rule::evidence_section {
                    let mut section_inner = part.into_inner();
                    let section_name = section_inner.next().unwrap().as_str().to_string();
                    // Skip colon
                    section_inner.next();
                    
                    if let Some(section_content) = section_inner.next() {
                        match section_content.as_rule() {
                            Rule::expression => {
                                let expr = Self::parse_expression(section_content)?;
                                sections.insert(section_name, EvidenceSection::Simple(expr));
                            }
                            Rule::evidence_subsection => {
                                let mut subsection_props = HashMap::new();
                                for prop in section_content.into_inner() {
                                    if prop.as_rule() == Rule::evidence_property {
                                        let mut prop_inner = prop.into_inner();
                                        let prop_name = prop_inner.next().unwrap().as_str().to_string();
                                        // Skip colon
                                        prop_inner.next();
                                        let prop_value = Self::parse_expression(prop_inner.next().unwrap())?;
                                        subsection_props.insert(prop_name, prop_value);
                                    }
                                }
                                sections.insert(section_name, EvidenceSection::Complex(subsection_props));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        
        Ok(EvidenceDeclaration { name, sections, span })
    }

    /// Parse expressions
    fn parse_expression(pair: pest::iterators::Pair<Rule>) -> Result<Expression> {
        match pair.as_rule() {
            Rule::expression => {
                let inner = pair.into_inner().next().unwrap();
                Self::parse_expression(inner)
            }
            Rule::primary_expr => {
                let inner = pair.into_inner().next().unwrap();
                Self::parse_primary_expression(inner)
            }
            Rule::binary_expr => {
                Self::parse_binary_expression(pair)
            }
            Rule::logical_or_expr |
            Rule::logical_and_expr |
            Rule::equality_expr |
            Rule::relational_expr |
            Rule::additive_expr |
            Rule::multiplicative_expr |
            Rule::power_expr |
            Rule::unary_expr |
            Rule::postfix_expr => {
                // For now, just parse as the first inner expression
                let inner = pair.into_inner().next().unwrap();
                Self::parse_expression(inner)
            }
            _ => Self::parse_primary_expression(pair)
        }
    }

    /// Parse primary expressions
    fn parse_primary_expression(pair: pest::iterators::Pair<Rule>) -> Result<Expression> {
        let span = Self::pair_to_span(&pair);
        
        match pair.as_rule() {
            Rule::identifier => Ok(Expression::Identifier(pair.as_str().to_string(), span)),
            Rule::string_literal => {
                let s = pair.as_str().trim_matches('"').to_string();
                Ok(Expression::StringLiteral(s, span))
            }
            Rule::number_literal => {
                let n = pair.as_str().parse::<f64>()?;
                Ok(Expression::NumberLiteral(n, span))
            }
            Rule::boolean_literal => {
                let b = pair.as_str() == "true";
                Ok(Expression::BooleanLiteral(b, span))
            }
            Rule::range_expr => {
                let mut inner = pair.into_inner();
                let start = Self::parse_expression(inner.next().unwrap())?;
                let end = Self::parse_expression(inner.next().unwrap())?;
                Ok(Expression::Range(Box::new(start), Box::new(end), span))
            }
            Rule::list_literal => {
                let mut items = Vec::new();
                for item in pair.into_inner() {
                    if item.as_rule() == Rule::expression {
                        items.push(Self::parse_expression(item)?);
                    }
                }
                Ok(Expression::List(items, span))
            }
            Rule::dict_literal => {
                let mut items = HashMap::new();
                for pair_item in pair.into_inner() {
                    if pair_item.as_rule() == Rule::dict_pair {
                        let mut pair_inner = pair_item.into_inner();
                        let key = match pair_inner.next().unwrap().as_rule() {
                            Rule::identifier => pair_inner.as_str().to_string(),
                            Rule::string_literal => pair_inner.as_str().trim_matches('"').to_string(),
                            _ => "unknown".to_string(),
                        };
                        // Skip colon
                        pair_inner.next();
                        let value = Self::parse_expression(pair_inner.next().unwrap())?;
                        items.insert(key, value);
                    }
                }
                Ok(Expression::Dictionary(items, span))
            }
            Rule::function_call => {
                let mut inner = pair.into_inner();
                let function_name = inner.next().unwrap().as_str().to_string();
                
                let mut arguments = Vec::new();
                for arg in inner {
                    if arg.as_rule() == Rule::expression {
                        arguments.push(Self::parse_expression(arg)?);
                    }
                }
                
                Ok(Expression::FunctionCall(FunctionCall {
                    function: function_name,
                    arguments,
                    span,
                }))
            }
            Rule::paren_expr => {
                let inner = pair.into_inner().next().unwrap();
                Self::parse_expression(inner)
            }
            _ => Err(anyhow!("Unexpected primary expression: {:?}", pair.as_rule())),
        }
    }

    /// Parse binary expressions (simplified for now)
    fn parse_binary_expression(pair: pest::iterators::Pair<Rule>) -> Result<Expression> {
        // For now, just parse the first expression
        let inner = pair.into_inner().next().unwrap();
        Self::parse_expression(inner)
    }

    /// Parse remaining complex declarations (simplified implementations)
    fn parse_metacognitive_declaration(pair: pest::iterators::Pair<Rule>) -> Result<MetacognitiveDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip keyword
        inner.next();
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(MetacognitiveDeclaration {
            name,
            sections: HashMap::new(),
            span,
        })
    }

    fn parse_temporal_declaration(pair: pest::iterators::Pair<Rule>) -> Result<TemporalDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip keyword
        inner.next();
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(TemporalDeclaration {
            name,
            sections: HashMap::new(),
            span,
        })
    }

    fn parse_bayesian_network_declaration(pair: pest::iterators::Pair<Rule>) -> Result<BayesianNetworkDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip keyword
        inner.next();
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(BayesianNetworkDeclaration {
            name,
            nodes: Vec::new(),
            edges: Vec::new(),
            optimization_targets: Vec::new(),
            span,
        })
    }

    fn parse_sensor_fusion_declaration(pair: pest::iterators::Pair<Rule>) -> Result<SensorFusionDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip keyword
        inner.next();
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(SensorFusionDeclaration {
            name,
            primary_sensors: Vec::new(),
            secondary_sensors: Vec::new(),
            fusion_strategy: HashMap::new(),
            calibration: HashMap::new(),
            span,
        })
    }

    fn parse_fuzzy_system_declaration(pair: pest::iterators::Pair<Rule>) -> Result<FuzzySystemDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        // Skip keyword
        inner.next();
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(FuzzySystemDeclaration {
            name,
            membership_functions: HashMap::new(),
            fuzzy_rules: Vec::new(),
            defuzzification: HashMap::new(),
            span,
        })
    }

    // Simplified implementations for remaining declarations
    fn parse_bayesian_update_declaration(pair: pest::iterators::Pair<Rule>) -> Result<BayesianUpdateDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(BayesianUpdateDeclaration {
            name,
            update_strategy: "default".to_string(),
            convergence_criteria: Expression::NumberLiteral(0.001, span.clone()),
            max_iterations: Expression::NumberLiteral(1000.0, span.clone()),
            evidence_integration: HashMap::new(),
            network_structure_adaptation: HashMap::new(),
            uncertainty_quantification: HashMap::new(),
            span,
        })
    }

    fn parse_real_time_declaration(pair: pest::iterators::Pair<Rule>) -> Result<RealTimeDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(RealTimeDeclaration {
            name,
            architecture: "default".to_string(),
            latency_target: Expression::NumberLiteral(50.0, span.clone()),
            throughput_target: Expression::NumberLiteral(240.0, span.clone()),
            pipeline_stages: Vec::new(),
            performance_monitoring: HashMap::new(),
            span,
        })
    }

    fn parse_optimization_framework_declaration(pair: pest::iterators::Pair<Rule>) -> Result<OptimizationFrameworkDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(OptimizationFrameworkDeclaration {
            name,
            objective_functions: HashMap::new(),
            optimization_variables: HashMap::new(),
            optimization_methods: HashMap::new(),
            personalization: HashMap::new(),
            span,
        })
    }

    fn parse_genetic_optimization_declaration(pair: pest::iterators::Pair<Rule>) -> Result<GeneticOptimizationDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(GeneticOptimizationDeclaration {
            name,
            population_size: Expression::NumberLiteral(100.0, span.clone()),
            generations: Expression::NumberLiteral(500.0, span.clone()),
            selection_method: "tournament".to_string(),
            crossover_method: "simulated_binary".to_string(),
            mutation_method: "polynomial".to_string(),
            genotype_representation: HashMap::new(),
            fitness_evaluation: HashMap::new(),
            evolution_strategies: HashMap::new(),
            convergence_acceleration: HashMap::new(),
            span,
        })
    }

    fn parse_analysis_workflow_declaration(pair: pest::iterators::Pair<Rule>) -> Result<AnalysisWorkflowDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(AnalysisWorkflowDeclaration {
            name,
            stages: Vec::new(),
            span,
        })
    }

    fn parse_validation_framework_declaration(pair: pest::iterators::Pair<Rule>) -> Result<ValidationFrameworkDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(ValidationFrameworkDeclaration {
            name,
            ground_truth_comparison: HashMap::new(),
            cross_validation_strategy: HashMap::new(),
            uncertainty_validation: HashMap::new(),
            performance_metrics: HashMap::new(),
            automated_validation_pipeline: HashMap::new(),
            span,
        })
    }

    fn parse_pattern_registry_declaration(pair: pest::iterators::Pair<Rule>) -> Result<PatternRegistryDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(PatternRegistryDeclaration {
            name,
            categories: HashMap::new(),
            pattern_matching: HashMap::new(),
            adaptation_learning: HashMap::new(),
            span,
        })
    }

    fn parse_orchestration_declaration(pair: pest::iterators::Pair<Rule>) -> Result<OrchestrationDeclaration> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        inner.next(); // Skip keyword
        let name = inner.next().unwrap().as_str().to_string();
        
        Ok(OrchestrationDeclaration {
            name,
            sections: HashMap::new(),
            span,
        })
    }

    // Simplified control flow parsing
    fn parse_assignment(pair: pest::iterators::Pair<Rule>) -> Result<Assignment> {
        let span = Self::pair_to_span(&pair);
        let mut inner = pair.into_inner();
        
        let target = inner.next().unwrap().as_str().to_string();
        let operator = AssignmentOperator::Assign; // Simplified
        let value = Self::parse_expression(inner.next().unwrap())?;
        
        Ok(Assignment { target, operator, value, span })
    }

    fn parse_if_statement(pair: pest::iterators::Pair<Rule>) -> Result<IfStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(IfStatement {
            conditions: Vec::new(),
            else_block: None,
            span,
        })
    }

    fn parse_for_statement(pair: pest::iterators::Pair<Rule>) -> Result<ForStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(ForStatement {
            variable: "item".to_string(),
            iterable: Expression::Identifier("items".to_string(), span.clone()),
            body: BlockStatement { statements: Vec::new(), span: span.clone() },
            span,
        })
    }

    fn parse_while_statement(pair: pest::iterators::Pair<Rule>) -> Result<WhileStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(WhileStatement {
            condition: Expression::BooleanLiteral(true, span.clone()),
            body: BlockStatement { statements: Vec::new(), span: span.clone() },
            span,
        })
    }

    fn parse_within_statement(pair: pest::iterators::Pair<Rule>) -> Result<WithinStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(WithinStatement {
            target: Expression::Identifier("data".to_string(), span.clone()),
            alias: None,
            body: WithinBody { contents: Vec::new(), span: span.clone() },
            span,
        })
    }

    fn parse_try_statement(pair: pest::iterators::Pair<Rule>) -> Result<TryStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(TryStatement {
            try_block: BlockStatement { statements: Vec::new(), span: span.clone() },
            catch_blocks: Vec::new(),
            finally_block: None,
            span,
        })
    }

    fn parse_return_statement(pair: pest::iterators::Pair<Rule>) -> Result<ReturnStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(ReturnStatement { value: None, span })
    }

    fn parse_import_statement(pair: pest::iterators::Pair<Rule>) -> Result<ImportStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(ImportStatement {
            import_type: ImportType::Simple { module: "module".to_string() },
            span,
        })
    }

    fn parse_block_statement(pair: pest::iterators::Pair<Rule>) -> Result<BlockStatement> {
        let span = Self::pair_to_span(&pair);
        let mut statements = Vec::new();
        
        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::statement {
                statements.push(Self::parse_statement(inner_pair)?);
            }
        }
        
        Ok(BlockStatement { statements, span })
    }

    fn parse_support_statement(pair: pest::iterators::Pair<Rule>) -> Result<SupportStatement> {
        let span = Self::pair_to_span(&pair);
        Ok(SupportStatement {
            motion: "motion".to_string(),
            modifier: None,
            span,
        })
    }

    fn parse_causal_chain_declaration(pair: pest::iterators::Pair<Rule>) -> Result<CausalChainDeclaration> {
        let span = Self::pair_to_span(&pair);
        Ok(CausalChainDeclaration {
            sequence: Vec::new(),
            properties: HashMap::new(),
            span,
        })
    }

    /// Convert pest pair to span
    fn pair_to_span(pair: &pest::iterators::Pair<Rule>) -> Span {
        let (start_line, start_col) = pair.line_col();
        Span {
            start: Position {
                line: start_line,
                column: start_col,
                file: None,
            },
            end: Position {
                line: start_line,
                column: start_col + pair.as_str().len(),
                file: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_variable() {
        let source = r#"item temperature = 23.5"#;
        let result = TurbulanceParser::parse_program(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_simple_function() {
        let source = r#"
        funxn calculate_average(numbers):
            item sum = 0
            return sum
        "#;
        let result = TurbulanceParser::parse_program(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_proposition() {
        let source = r#"
        proposition TestAnalysis:
            context data = load_data("test.csv")
            motion TestMotion("Test motion description"):
                threshold: 0.8
        "#;
        let result = TurbulanceParser::parse_program(source);
        assert!(result.is_ok());
    }
} 