use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Position information for error reporting and debugging
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub file: Option<String>,
}

/// Span information covering a range in source code
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

/// Main program node - root of the AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub statements: Vec<Statement>,
    pub span: Span,
}

/// All possible statement types in Turbulance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    VariableDeclaration(VariableDeclaration),
    FunctionDeclaration(FunctionDeclaration),
    PropositionDeclaration(PropositionDeclaration),
    MotionDeclaration(MotionDeclaration),
    EvidenceDeclaration(EvidenceDeclaration),
    MetacognitiveDeclaration(MetacognitiveDeclaration),
    TemporalDeclaration(TemporalDeclaration),
    BayesianNetworkDeclaration(BayesianNetworkDeclaration),
    SensorFusionDeclaration(SensorFusionDeclaration),
    FuzzySystemDeclaration(FuzzySystemDeclaration),
    BayesianUpdateDeclaration(BayesianUpdateDeclaration),
    RealTimeDeclaration(RealTimeDeclaration),
    OptimizationFrameworkDeclaration(OptimizationFrameworkDeclaration),
    GeneticOptimizationDeclaration(GeneticOptimizationDeclaration),
    AnalysisWorkflowDeclaration(AnalysisWorkflowDeclaration),
    ValidationFrameworkDeclaration(ValidationFrameworkDeclaration),
    PatternRegistryDeclaration(PatternRegistryDeclaration),
    OrchestrationDeclaration(OrchestrationDeclaration),
    Assignment(Assignment),
    Expression(Expression),
    If(IfStatement),
    For(ForStatement),
    While(WhileStatement),
    Within(WithinStatement),
    Try(TryStatement),
    Return(ReturnStatement),
    Import(ImportStatement),
    Block(BlockStatement),
    Support(SupportStatement),
    CausalChain(CausalChainDeclaration),
}

/// Variable declaration: item name = value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableDeclaration {
    pub name: String,
    pub type_annotation: Option<TypeExpression>,
    pub initializer: Expression,
    pub span: Span,
}

/// Function declaration: funxn name(params) -> return_type: { body }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<TypeExpression>,
    pub body: BlockStatement,
    pub span: Span,
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<TypeExpression>,
    pub span: Span,
}

/// Type expressions for type annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpression {
    Named {
        name: String,
        generics: Vec<TypeExpression>,
    },
    Optional(Box<TypeExpression>),
    Function {
        parameters: Vec<TypeExpression>,
        return_type: Box<TypeExpression>,
    },
}

/// Proposition declaration for hypothesis testing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropositionDeclaration {
    pub name: String,
    pub contexts: Vec<ContextDeclaration>,
    pub motions: Vec<MotionDeclaration>,
    pub within_blocks: Vec<WithinStatement>,
    pub span: Span,
}

/// Context declaration within propositions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextDeclaration {
    pub name: String,
    pub value: Expression,
    pub span: Span,
}

/// Motion declaration - sub-hypotheses within propositions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionDeclaration {
    pub name: String,
    pub description: String,
    pub properties: HashMap<String, Expression>,
    pub span: Span,
}

/// Evidence declaration for data collection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceDeclaration {
    pub name: String,
    pub sections: HashMap<String, EvidenceSection>,
    pub span: Span,
}

/// Section within evidence declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceSection {
    Simple(Expression),
    Complex(HashMap<String, Expression>),
}

/// Metacognitive declaration for self-reflective analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetacognitiveDeclaration {
    pub name: String,
    pub sections: HashMap<String, MetacognitiveSection>,
    pub span: Span,
}

/// Section within metacognitive declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetacognitiveSection {
    Simple(Expression),
    Complex(HashMap<String, Expression>),
}

/// Temporal analysis declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalDeclaration {
    pub name: String,
    pub sections: HashMap<String, TemporalSection>,
    pub span: Span,
}

/// Section within temporal declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalSection {
    Simple(Expression),
    Complex(HashMap<String, Expression>),
}

/// Bayesian network declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianNetworkDeclaration {
    pub name: String,
    pub nodes: Vec<BayesianNode>,
    pub edges: Vec<BayesianEdge>,
    pub optimization_targets: Vec<OptimizationTarget>,
    pub span: Span,
}

/// Node in Bayesian network
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianNode {
    pub name: String,
    pub node_type: String,
    pub properties: HashMap<String, Expression>,
    pub span: Span,
}

/// Edge in Bayesian network
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianEdge {
    pub from: String,
    pub to: String,
    pub relationship_type: String,
    pub strength: Expression,
    pub fuzziness: Option<Expression>,
    pub span: Span,
}

/// Optimization target
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationTarget {
    pub target_type: String, // maximize, minimize, balance
    pub expression: Expression,
    pub span: Span,
}

/// Sensor fusion declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorFusionDeclaration {
    pub name: String,
    pub primary_sensors: Vec<SensorDefinition>,
    pub secondary_sensors: Vec<SensorDefinition>,
    pub fusion_strategy: HashMap<String, Expression>,
    pub calibration: HashMap<String, Expression>,
    pub span: Span,
}

/// Sensor definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorDefinition {
    pub name: String,
    pub sensor_type: String,
    pub properties: HashMap<String, Expression>,
    pub span: Span,
}

/// Fuzzy system declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzySystemDeclaration {
    pub name: String,
    pub membership_functions: HashMap<String, FuzzyMembershipFunction>,
    pub fuzzy_rules: Vec<FuzzyRule>,
    pub defuzzification: HashMap<String, Expression>,
    pub span: Span,
}

/// Fuzzy membership function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyMembershipFunction {
    pub name: String,
    pub function_type: String, // triangular, trapezoidal, gaussian, sigmoid
    pub parameters: Vec<Expression>,
    pub span: Span,
}

/// Fuzzy rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub description: String,
    pub conditions: Vec<FuzzyCondition>,
    pub conclusions: Vec<FuzzyConclusion>,
    pub weight: Option<Expression>,
    pub span: Span,
}

/// Fuzzy condition in rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyCondition {
    pub variable: String,
    pub linguistic_value: String,
    pub span: Span,
}

/// Fuzzy conclusion in rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyConclusion {
    pub variable: String,
    pub action: String,
    pub value: Expression,
    pub span: Span,
}

/// Bayesian update declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BayesianUpdateDeclaration {
    pub name: String,
    pub update_strategy: String,
    pub convergence_criteria: Expression,
    pub max_iterations: Expression,
    pub evidence_integration: HashMap<String, Expression>,
    pub network_structure_adaptation: HashMap<String, Expression>,
    pub uncertainty_quantification: HashMap<String, Expression>,
    pub span: Span,
}

/// Real-time analysis declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealTimeDeclaration {
    pub name: String,
    pub architecture: String,
    pub latency_target: Expression,
    pub throughput_target: Expression,
    pub pipeline_stages: Vec<PipelineStage>,
    pub performance_monitoring: HashMap<String, Expression>,
    pub span: Span,
}

/// Pipeline stage in real-time analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    pub stage_type: String,
    pub properties: HashMap<String, Expression>,
    pub output: String,
    pub span: Span,
}

/// Optimization framework declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationFrameworkDeclaration {
    pub name: String,
    pub objective_functions: HashMap<String, Expression>,
    pub optimization_variables: HashMap<String, OptimizationVariable>,
    pub optimization_methods: HashMap<String, Expression>,
    pub personalization: HashMap<String, Expression>,
    pub span: Span,
}

/// Optimization variable
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationVariable {
    pub name: String,
    pub variable_type: String, // continuous, discrete, categorical
    pub range: Option<(Expression, Expression)>,
    pub constraints: Vec<Expression>,
    pub span: Span,
}

/// Genetic optimization declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneticOptimizationDeclaration {
    pub name: String,
    pub population_size: Expression,
    pub generations: Expression,
    pub selection_method: String,
    pub crossover_method: String,
    pub mutation_method: String,
    pub genotype_representation: HashMap<String, Expression>,
    pub fitness_evaluation: HashMap<String, Expression>,
    pub evolution_strategies: HashMap<String, Expression>,
    pub convergence_acceleration: HashMap<String, Expression>,
    pub span: Span,
}

/// Analysis workflow declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisWorkflowDeclaration {
    pub name: String,
    pub stages: Vec<WorkflowStage>,
    pub span: Span,
}

/// Workflow stage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowStage {
    pub name: String,
    pub stage_type: String,
    pub operations: Vec<Statement>,
    pub span: Span,
}

/// Validation framework declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationFrameworkDeclaration {
    pub name: String,
    pub ground_truth_comparison: HashMap<String, Expression>,
    pub cross_validation_strategy: HashMap<String, Expression>,
    pub uncertainty_validation: HashMap<String, Expression>,
    pub performance_metrics: HashMap<String, Expression>,
    pub automated_validation_pipeline: HashMap<String, Expression>,
    pub span: Span,
}

/// Pattern registry declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternRegistryDeclaration {
    pub name: String,
    pub categories: HashMap<String, Vec<PatternDefinition>>,
    pub pattern_matching: HashMap<String, Expression>,
    pub adaptation_learning: HashMap<String, Expression>,
    pub span: Span,
}

/// Pattern definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternDefinition {
    pub name: String,
    pub pattern_type: String,
    pub properties: HashMap<String, Expression>,
    pub span: Span,
}

/// Orchestration declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrchestrationDeclaration {
    pub name: String,
    pub sections: HashMap<String, OrchestrationSection>,
    pub span: Span,
}

/// Section within orchestration declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrchestrationSection {
    Simple(Expression),
    Complex(HashMap<String, Expression>),
}

/// Assignment statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    pub target: String,
    pub operator: AssignmentOperator,
    pub value: Expression,
    pub span: Span,
}

/// Assignment operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssignmentOperator {
    Assign,      // =
    AddAssign,   // +=
    SubAssign,   // -=
    MulAssign,   // *=
    DivAssign,   // /=
}

/// If statement (given/otherwise in Turbulance)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IfStatement {
    pub conditions: Vec<(Expression, BlockStatement)>,
    pub else_block: Option<BlockStatement>,
    pub span: Span,
}

/// For loop statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForStatement {
    pub variable: String,
    pub iterable: Expression,
    pub body: BlockStatement,
    pub span: Span,
}

/// While loop statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhileStatement {
    pub condition: Expression,
    pub body: BlockStatement,
    pub span: Span,
}

/// Within statement for pattern-based iteration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithinStatement {
    pub target: Expression,
    pub alias: Option<String>,
    pub body: WithinBody,
    pub span: Span,
}

/// Body of within statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WithinBody {
    pub contents: Vec<WithinContent>,
    pub span: Span,
}

/// Content within a within statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WithinContent {
    FuzzyEvaluate(FuzzyEvaluateBlock),
    Biomechanical(BiomechanicalBlock),
    CausalInference(CausalInferenceBlock),
    TemporalAnalysis(TemporalAnalysisBlock),
    PatternMatching(PatternMatchingBlock),
    AdvancedAnalysis(AdvancedAnalysisBlock),
    Statement(Box<Statement>),
}

/// Fuzzy evaluate block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyEvaluateBlock {
    pub name: String,
    pub body: BlockStatement,
    pub span: Span,
}

/// Biomechanical analysis block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BiomechanicalBlock {
    pub name: String,
    pub body: BlockStatement,
    pub span: Span,
}

/// Causal inference block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalInferenceBlock {
    pub name: String,
    pub body: BlockStatement,
    pub span: Span,
}

/// Temporal analysis block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalAnalysisBlock {
    pub name: String,
    pub body: BlockStatement,
    pub span: Span,
}

/// Pattern matching block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternMatchingBlock {
    pub condition: Expression,
    pub body: BlockStatement,
    pub span: Span,
}

/// Advanced analysis block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdvancedAnalysisBlock {
    pub analysis_type: String,
    pub name: String,
    pub body: BlockStatement,
    pub span: Span,
}

/// Try statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TryStatement {
    pub try_block: BlockStatement,
    pub catch_blocks: Vec<CatchBlock>,
    pub finally_block: Option<BlockStatement>,
    pub span: Span,
}

/// Catch block
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatchBlock {
    pub exception_type: String,
    pub variable: String,
    pub body: BlockStatement,
    pub span: Span,
}

/// Return statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReturnStatement {
    pub value: Option<Expression>,
    pub span: Span,
}

/// Import statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportStatement {
    pub import_type: ImportType,
    pub span: Span,
}

/// Types of import statements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportType {
    Simple { module: String },
    From { module: String, items: Vec<String>, alias: Option<String> },
}

/// Block statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockStatement {
    pub statements: Vec<Statement>,
    pub span: Span,
}

/// Support statement for evidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportStatement {
    pub motion: String,
    pub modifier: Option<SupportModifier>,
    pub span: Span,
}

/// Support modifiers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SupportModifier {
    WithConfidence(Expression),
    WithWeight(Expression),
    WithEvidence(HashMap<String, Expression>),
}

/// Causal chain declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalChainDeclaration {
    pub sequence: Vec<String>,
    pub properties: HashMap<String, Expression>,
    pub span: Span,
}

/// All expression types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    Identifier(String, Span),
    StringLiteral(String, Span),
    NumberLiteral(f64, Span),
    BooleanLiteral(bool, Span),
    Range(Box<Expression>, Box<Expression>, Span),
    List(Vec<Expression>, Span),
    Dictionary(HashMap<String, Expression>, Span),
    FunctionCall(FunctionCall),
    MemberAccess(MemberAccess),
    BinaryOp(BinaryOperation),
    UnaryOp(UnaryOperation),
}

/// Function call expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub function: String,
    pub arguments: Vec<Expression>,
    pub span: Span,
}

/// Member access expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemberAccess {
    pub object: Box<Expression>,
    pub member: String,
    pub span: Span,
}

/// Binary operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryOperation {
    pub left: Box<Expression>,
    pub operator: BinaryOperator,
    pub right: Box<Expression>,
    pub span: Span,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Sub, Mul, Div, Mod, Pow,
    // Comparison
    Eq, Ne, Lt, Gt, Le, Ge,
    // Logical
    And, Or,
    // Membership
    In,
}

/// Unary operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnaryOperation {
    pub operator: UnaryOperator,
    pub operand: Box<Expression>,
    pub span: Span,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
}

// Display implementations for debugging
impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Identifier(name, _) => write!(f, "{}", name),
            Expression::StringLiteral(s, _) => write!(f, "\"{}\"", s),
            Expression::NumberLiteral(n, _) => write!(f, "{}", n),
            Expression::BooleanLiteral(b, _) => write!(f, "{}", b),
            Expression::Range(start, end, _) => write!(f, "{}..{}", start, end),
            Expression::List(items, _) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            },
            Expression::Dictionary(items, _) => {
                write!(f, "{{")?;
                for (i, (key, value)) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", key, value)?;
                }
                write!(f, "}}")
            },
            Expression::FunctionCall(call) => {
                write!(f, "{}(", call.function)?;
                for (i, arg) in call.arguments.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            },
            Expression::MemberAccess(access) => write!(f, "{}.{}", access.object, access.member),
            Expression::BinaryOp(op) => write!(f, "({} {:?} {})", op.left, op.operator, op.right),
            Expression::UnaryOp(op) => write!(f, "({:?}{})", op.operator, op.operand),
        }
    }
} 