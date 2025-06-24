use crate::turbulance::ast::*;
use std::collections::HashMap;
use anyhow::{Result, anyhow};

/// Code generator that converts Turbulance AST to executable Python code
pub struct PythonCodeGenerator {
    /// Indentation level for proper formatting
    indent_level: usize,
    /// Generated imports
    imports: Vec<String>,
    /// Generated class definitions
    classes: Vec<String>,
    /// Generated function definitions
    functions: Vec<String>,
    /// Main execution code
    main_code: Vec<String>,
}

impl PythonCodeGenerator {
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            imports: vec![
                "import numpy as np".to_string(),
                "import pandas as pd".to_string(),
                "from typing import Dict, List, Optional, Union, Any".to_string(),
                "from dataclasses import dataclass".to_string(),
                "import asyncio".to_string(),
                "from abc import ABC, abstractmethod".to_string(),
                "".to_string(),
                "# Moriarty framework imports".to_string(),
                "from src.core.pose.pose_detector import PoseDetector".to_string(),
                "from src.core.dynamics.dynamics_analyzer import DynamicsAnalyzer".to_string(),
                "from src.core.motion.movement_tracker import MovementTracker".to_string(),
                "from src.utils.visualization import Visualizer".to_string(),
                "from src.pipeline import VideoPipeline".to_string(),
                "".to_string(),
                "# Scientific computing imports".to_string(),
                "from scipy import stats".to_string(),
                "from sklearn.ensemble import IsolationForest".to_string(),
                "import cv2".to_string(),
                "".to_string(),
            ],
            classes: Vec::new(),
            functions: Vec::new(),
            main_code: Vec::new(),
        }
    }

    /// Generate Python code from Turbulance program
    pub fn generate(&mut self, program: &Program) -> Result<String> {
        // Add runtime framework first
        self.add_runtime_framework();
        
        // Process all statements
        for statement in &program.statements {
            self.generate_statement(statement)?;
        }

        // Combine all parts
        let mut output = Vec::new();
        output.extend(self.imports.clone());
        output.extend(self.classes.clone());
        output.extend(self.functions.clone());
        
        // Add main execution block
        if !self.main_code.is_empty() {
            output.push("".to_string());
            output.push("if __name__ == '__main__':".to_string());
            for line in &self.main_code {
                output.push(format!("    {}", line));
            }
        }

        Ok(output.join("\n"))
    }

    /// Add the runtime framework for Turbulance execution
    fn add_runtime_framework(&mut self) {
        let framework_code = r#"
# Turbulance Runtime Framework
class TurbulanceRuntime:
    """Main runtime for executing Turbulance programs"""
    
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.propositions = {}
        self.evidence_store = {}
        self.bayesian_networks = {}
        self.fuzzy_systems = {}
        self.moriarty_pipeline = None
        
    def initialize_moriarty(self, config=None):
        """Initialize the Moriarty pipeline"""
        self.moriarty_pipeline = VideoPipeline(memory_limit=0.4)
        
    def set_variable(self, name: str, value: Any):
        """Set a variable in the runtime"""
        self.variables[name] = value
        
    def get_variable(self, name: str) -> Any:
        """Get a variable from the runtime"""
        return self.variables.get(name)

@dataclass
class Evidence:
    """Evidence container for Turbulance analysis"""
    name: str
    data: Any
    confidence: float = 1.0
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Motion:
    """Motion (sub-hypothesis) in Turbulance propositions"""
    name: str
    description: str
    properties: Dict[str, Any]
    evidence: List[Evidence]
    support_level: float = 0.0
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

class Proposition:
    """Proposition for hypothesis testing in Turbulance"""
    
    def __init__(self, name: str):
        self.name = name
        self.contexts = {}
        self.motions = {}
        self.evidence = []
        self.current_support = 0.0
        
    def add_context(self, name: str, value: Any):
        """Add context to the proposition"""
        self.contexts[name] = value
        
    def add_motion(self, motion: Motion):
        """Add a motion to the proposition"""
        self.motions[motion.name] = motion
        
    def support_motion(self, motion_name: str, evidence: Evidence, weight: float = 1.0):
        """Support a motion with evidence"""
        if motion_name in self.motions:
            self.motions[motion_name].evidence.append(evidence)
            self.motions[motion_name].support_level += evidence.confidence * weight
            self._update_overall_support()
    
    def _update_overall_support(self):
        """Update overall proposition support"""
        if self.motions:
            self.current_support = sum(m.support_level for m in self.motions.values()) / len(self.motions)

class BayesianNetwork:
    """Bayesian network for probabilistic reasoning"""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.evidence = {}
        
    def add_node(self, name: str, node_type: str, properties: Dict[str, Any]):
        """Add a node to the network"""
        self.nodes[name] = {
            'type': node_type,
            'properties': properties,
            'belief': 0.5  # Prior belief
        }
        
    def add_edge(self, from_node: str, to_node: str, strength: float, fuzziness: float = 0.0):
        """Add an edge between nodes"""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append({
            'to': to_node,
            'strength': strength,
            'fuzziness': fuzziness
        })
        
    def update_belief(self, node: str, evidence_value: float, confidence: float = 1.0):
        """Update belief in a node based on evidence"""
        if node in self.nodes:
            # Simple Bayesian update (simplified)
            prior = self.nodes[node]['belief']
            likelihood = evidence_value * confidence
            posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))
            self.nodes[node]['belief'] = posterior
            
            # Propagate to connected nodes
            self._propagate_belief(node)
            
    def _propagate_belief(self, from_node: str):
        """Propagate belief changes to connected nodes"""
        if from_node in self.edges:
            for edge in self.edges[from_node]:
                to_node = edge['to']
                strength = edge['strength']
                if to_node in self.nodes:
                    influence = self.nodes[from_node]['belief'] * strength
                    self.nodes[to_node]['belief'] = min(1.0, self.nodes[to_node]['belief'] + influence * 0.1)

class FuzzySystem:
    """Fuzzy logic system for handling uncertainty"""
    
    def __init__(self, name: str):
        self.name = name
        self.membership_functions = {}
        self.rules = []
        
    def add_membership_function(self, variable: str, name: str, func_type: str, params: List[float]):
        """Add a membership function"""
        if variable not in self.membership_functions:
            self.membership_functions[variable] = {}
        self.membership_functions[variable][name] = {
            'type': func_type,
            'params': params
        }
        
    def add_rule(self, conditions: List[tuple], conclusions: List[tuple], weight: float = 1.0):
        """Add a fuzzy rule"""
        self.rules.append({
            'conditions': conditions,  # [(variable, linguistic_value), ...]
            'conclusions': conclusions,  # [(variable, action, value), ...]
            'weight': weight
        })
        
    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Evaluate the fuzzy system"""
        results = {}
        
        for rule in self.rules:
            # Calculate rule activation
            activation = 1.0
            for var, linguistic_value in rule['conditions']:
                if var in inputs:
                    membership = self._calculate_membership(var, linguistic_value, inputs[var])
                    activation = min(activation, membership)
            
            # Apply conclusions
            for var, action, value in rule['conclusions']:
                if var not in results:
                    results[var] = 0.0
                if action == 'is':
                    results[var] = max(results[var], activation * value)
                elif action == 'increased_by':
                    results[var] += activation * value
                elif action == 'reduced_by':
                    results[var] -= activation * value
                    
        return results
        
    def _calculate_membership(self, variable: str, linguistic_value: str, input_value: float) -> float:
        """Calculate membership degree"""
        if variable in self.membership_functions and linguistic_value in self.membership_functions[variable]:
            func = self.membership_functions[variable][linguistic_value]
            func_type = func['type']
            params = func['params']
            
            if func_type == 'triangular':
                a, b, c = params[:3]
                if input_value <= a or input_value >= c:
                    return 0.0
                elif input_value <= b:
                    return (input_value - a) / (b - a)
                else:
                    return (c - input_value) / (c - b)
            elif func_type == 'trapezoidal':
                a, b, c, d = params[:4]
                if input_value <= a or input_value >= d:
                    return 0.0
                elif input_value <= b:
                    return (input_value - a) / (b - a)
                elif input_value <= c:
                    return 1.0
                else:
                    return (d - input_value) / (d - c)
            elif func_type == 'gaussian':
                center, sigma = params[:2]
                return np.exp(-0.5 * ((input_value - center) / sigma) ** 2)
                
        return 0.0

class SensorFusion:
    """Multi-modal sensor fusion for comprehensive analysis"""
    
    def __init__(self, name: str):
        self.name = name
        self.sensors = {}
        self.fusion_strategy = {}
        
    def add_sensor(self, name: str, sensor_type: str, properties: Dict[str, Any]):
        """Add a sensor to the fusion system"""
        self.sensors[name] = {
            'type': sensor_type,
            'properties': properties,
            'data': None,
            'last_update': None
        }
        
    def update_sensor_data(self, sensor_name: str, data: Any):
        """Update data from a sensor"""
        if sensor_name in self.sensors:
            self.sensors[sensor_name]['data'] = data
            self.sensors[sensor_name]['last_update'] = pd.Timestamp.now()
            
    def fuse_data(self) -> Dict[str, Any]:
        """Fuse data from all sensors"""
        fused_result = {}
        
        # Simple fusion strategy (weighted average where applicable)
        for sensor_name, sensor_info in self.sensors.items():
            if sensor_info['data'] is not None:
                weight = sensor_info['properties'].get('weight', 1.0)
                fused_result[sensor_name] = {
                    'data': sensor_info['data'],
                    'weight': weight,
                    'timestamp': sensor_info['last_update']
                }
                
        return fused_result

class RealTimeAnalyzer:
    """Real-time analysis pipeline"""
    
    def __init__(self, name: str, latency_target: float = 50.0):
        self.name = name
        self.latency_target = latency_target
        self.pipeline_stages = []
        self.performance_metrics = {}
        
    def add_stage(self, stage_name: str, stage_func, properties: Dict[str, Any]):
        """Add a pipeline stage"""
        self.pipeline_stages.append({
            'name': stage_name,
            'function': stage_func,
            'properties': properties
        })
        
    async def process_stream(self, data_stream):
        """Process streaming data through the pipeline"""
        for data in data_stream:
            start_time = pd.Timestamp.now()
            
            result = data
            for stage in self.pipeline_stages:
                result = await self._execute_stage(stage, result)
                
            end_time = pd.Timestamp.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            self.performance_metrics['last_latency'] = latency
            self.performance_metrics['latency_target_met'] = latency <= self.latency_target
            
            yield result
            
    async def _execute_stage(self, stage: Dict[str, Any], data: Any) -> Any:
        """Execute a pipeline stage"""
        if asyncio.iscoroutinefunction(stage['function']):
            return await stage['function'](data)
        else:
            return stage['function'](data)

# Initialize global runtime
turbulance_runtime = TurbulanceRuntime()
"#;

        self.classes.push(framework_code.to_string());
    }

    /// Generate code for a statement
    fn generate_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::VariableDeclaration(var_decl) => self.generate_variable_declaration(var_decl),
            Statement::FunctionDeclaration(func_decl) => self.generate_function_declaration(func_decl),
            Statement::PropositionDeclaration(prop_decl) => self.generate_proposition_declaration(prop_decl),
            Statement::BayesianNetworkDeclaration(bn_decl) => self.generate_bayesian_network_declaration(bn_decl),
            Statement::SensorFusionDeclaration(sf_decl) => self.generate_sensor_fusion_declaration(sf_decl),
            Statement::FuzzySystemDeclaration(fs_decl) => self.generate_fuzzy_system_declaration(fs_decl),
            Statement::RealTimeDeclaration(rt_decl) => self.generate_real_time_declaration(rt_decl),
            Statement::Assignment(assignment) => self.generate_assignment(assignment),
            Statement::Expression(expr) => self.generate_expression_statement(expr),
            Statement::Within(within_stmt) => self.generate_within_statement(within_stmt),
            _ => {
                // For unimplemented statements, add a comment
                self.main_code.push(format!("# TODO: Implement {:?}", statement));
                Ok(())
            }
        }
    }

    /// Generate variable declaration
    fn generate_variable_declaration(&mut self, var_decl: &VariableDeclaration) -> Result<()> {
        let value_code = self.generate_expression_code(&var_decl.initializer)?;
        self.main_code.push(format!("turbulance_runtime.set_variable('{}', {})", var_decl.name, value_code));
        self.main_code.push(format!("{} = turbulance_runtime.get_variable('{}')", var_decl.name, var_decl.name));
        Ok(())
    }

    /// Generate function declaration
    fn generate_function_declaration(&mut self, func_decl: &FunctionDeclaration) -> Result<()> {
        let mut func_code = Vec::new();
        
        // Function signature
        let params: Vec<String> = func_decl.parameters.iter()
            .map(|p| p.name.clone())
            .collect();
        func_code.push(format!("def {}({}):", func_decl.name, params.join(", ")));
        
        // Function body
        self.indent_level += 1;
        for statement in &func_decl.body.statements {
            match statement {
                Statement::Return(ret_stmt) => {
                    if let Some(value) = &ret_stmt.value {
                        let return_code = self.generate_expression_code(value)?;
                        func_code.push(format!("{}return {}", self.indent(), return_code));
                    } else {
                        func_code.push(format!("{}return", self.indent()));
                    }
                }
                _ => {
                    // For now, add placeholder
                    func_code.push(format!("{}# TODO: Implement statement", self.indent()));
                }
            }
        }
        self.indent_level -= 1;
        
        if func_code.len() == 1 {
            func_code.push("    pass".to_string());
        }
        
        self.functions.push(func_code.join("\n"));
        Ok(())
    }

    /// Generate proposition declaration
    fn generate_proposition_declaration(&mut self, prop_decl: &PropositionDeclaration) -> Result<()> {
        let mut prop_code = Vec::new();
        
        // Create proposition
        prop_code.push(format!("{} = Proposition('{}')", prop_decl.name, prop_decl.name));
        
        // Add contexts
        for context in &prop_decl.contexts {
            let context_value = self.generate_expression_code(&context.value)?;
            prop_code.push(format!("{}.add_context('{}', {})", prop_decl.name, context.name, context_value));
        }
        
        // Add motions
        for motion in &prop_decl.motions {
            prop_code.push(format!("{}_motion = Motion('{}', '{}', {{}})", 
                motion.name, motion.name, motion.description));
            
            // Add motion properties
            for (prop_name, prop_value) in &motion.properties {
                let value_code = self.generate_expression_code(prop_value)?;
                prop_code.push(format!("{}_motion.properties['{}'] = {}", 
                    motion.name, prop_name, value_code));
            }
            
            prop_code.push(format!("{}.add_motion({}_motion)", prop_decl.name, motion.name));
        }
        
        // Add to runtime
        prop_code.push(format!("turbulance_runtime.propositions['{}'] = {}", prop_decl.name, prop_decl.name));
        
        self.main_code.extend(prop_code);
        Ok(())
    }

    /// Generate Bayesian network declaration
    fn generate_bayesian_network_declaration(&mut self, bn_decl: &BayesianNetworkDeclaration) -> Result<()> {
        let mut bn_code = Vec::new();
        
        // Create network
        bn_code.push(format!("{} = BayesianNetwork('{}')", bn_decl.name, bn_decl.name));
        
        // Add nodes
        for node in &bn_decl.nodes {
            let mut props_code = Vec::new();
            for (key, value) in &node.properties {
                let value_code = self.generate_expression_code(value)?;
                props_code.push(format!("'{}': {}", key, value_code));
            }
            bn_code.push(format!("{}.add_node('{}', '{}', {{{}}})", 
                bn_decl.name, node.name, node.node_type, props_code.join(", ")));
        }
        
        // Add edges
        for edge in &bn_decl.edges {
            let strength_code = self.generate_expression_code(&edge.strength)?;
            let fuzziness_code = if let Some(fuzz) = &edge.fuzziness {
                self.generate_expression_code(fuzz)?
            } else {
                "0.0".to_string()
            };
            bn_code.push(format!("{}.add_edge('{}', '{}', {}, {})", 
                bn_decl.name, edge.from, edge.to, strength_code, fuzziness_code));
        }
        
        // Add to runtime
        bn_code.push(format!("turbulance_runtime.bayesian_networks['{}'] = {}", bn_decl.name, bn_decl.name));
        
        self.main_code.extend(bn_code);
        Ok(())
    }

    /// Generate sensor fusion declaration
    fn generate_sensor_fusion_declaration(&mut self, sf_decl: &SensorFusionDeclaration) -> Result<()> {
        let mut sf_code = Vec::new();
        
        // Create sensor fusion
        sf_code.push(format!("{} = SensorFusion('{}')", sf_decl.name, sf_decl.name));
        
        // Add primary sensors
        for sensor in &sf_decl.primary_sensors {
            let mut props_code = Vec::new();
            for (key, value) in &sensor.properties {
                let value_code = self.generate_expression_code(value)?;
                props_code.push(format!("'{}': {}", key, value_code));
            }
            sf_code.push(format!("{}.add_sensor('{}', '{}', {{{}}})", 
                sf_decl.name, sensor.name, sensor.sensor_type, props_code.join(", ")));
        }
        
        // Add secondary sensors
        for sensor in &sf_decl.secondary_sensors {
            let mut props_code = Vec::new();
            for (key, value) in &sensor.properties {
                let value_code = self.generate_expression_code(value)?;
                props_code.push(format!("'{}': {}", key, value_code));
            }
            sf_code.push(format!("{}.add_sensor('{}', '{}', {{{}}})", 
                sf_decl.name, sensor.name, sensor.sensor_type, props_code.join(", ")));
        }
        
        self.main_code.extend(sf_code);
        Ok(())
    }

    /// Generate fuzzy system declaration
    fn generate_fuzzy_system_declaration(&mut self, fs_decl: &FuzzySystemDeclaration) -> Result<()> {
        let mut fs_code = Vec::new();
        
        // Create fuzzy system
        fs_code.push(format!("{} = FuzzySystem('{}')", fs_decl.name, fs_decl.name));
        
        // Add membership functions
        for (var_name, membership_func) in &fs_decl.membership_functions {
            let mut params_code = Vec::new();
            for param in &membership_func.parameters {
                params_code.push(self.generate_expression_code(param)?);
            }
            fs_code.push(format!("{}.add_membership_function('{}', '{}', '{}', [{}])", 
                fs_decl.name, var_name, membership_func.name, membership_func.function_type, 
                params_code.join(", ")));
        }
        
        // Add fuzzy rules
        for rule in &fs_decl.fuzzy_rules {
            let conditions: Vec<String> = rule.conditions.iter()
                .map(|c| format!("('{}', '{}')", c.variable, c.linguistic_value))
                .collect();
            let conclusions: Vec<String> = rule.conclusions.iter()
                .map(|c| {
                    let value_code = self.generate_expression_code(&c.value).unwrap_or_else(|_| "0.0".to_string());
                    format!("('{}', '{}', {})", c.variable, c.action, value_code)
                })
                .collect();
            let weight_code = if let Some(weight) = &rule.weight {
                self.generate_expression_code(weight)?
            } else {
                "1.0".to_string()
            };
            fs_code.push(format!("{}.add_rule([{}], [{}], {})", 
                fs_decl.name, conditions.join(", "), conclusions.join(", "), weight_code));
        }
        
        // Add to runtime
        fs_code.push(format!("turbulance_runtime.fuzzy_systems['{}'] = {}", fs_decl.name, fs_decl.name));
        
        self.main_code.extend(fs_code);
        Ok(())
    }

    /// Generate real-time declaration
    fn generate_real_time_declaration(&mut self, rt_decl: &RealTimeDeclaration) -> Result<()> {
        let mut rt_code = Vec::new();
        
        let latency_code = self.generate_expression_code(&rt_decl.latency_target)?;
        rt_code.push(format!("{} = RealTimeAnalyzer('{}', {})", 
            rt_decl.name, rt_decl.name, latency_code));
        
        // Add pipeline stages
        for stage in &rt_decl.pipeline_stages {
            rt_code.push(format!("# TODO: Add pipeline stage '{}'", stage.name));
        }
        
        self.main_code.extend(rt_code);
        Ok(())
    }

    /// Generate assignment
    fn generate_assignment(&mut self, assignment: &Assignment) -> Result<()> {
        let value_code = self.generate_expression_code(&assignment.value)?;
        match assignment.operator {
            AssignmentOperator::Assign => {
                self.main_code.push(format!("{} = {}", assignment.target, value_code));
                self.main_code.push(format!("turbulance_runtime.set_variable('{}', {})", assignment.target, assignment.target));
            }
            AssignmentOperator::AddAssign => {
                self.main_code.push(format!("{} += {}", assignment.target, value_code));
            }
            AssignmentOperator::SubAssign => {
                self.main_code.push(format!("{} -= {}", assignment.target, value_code));
            }
            AssignmentOperator::MulAssign => {
                self.main_code.push(format!("{} *= {}", assignment.target, value_code));
            }
            AssignmentOperator::DivAssign => {
                self.main_code.push(format!("{} /= {}", assignment.target, value_code));
            }
        }
        Ok(())
    }

    /// Generate expression statement
    fn generate_expression_statement(&mut self, expr: &Expression) -> Result<()> {
        let expr_code = self.generate_expression_code(expr)?;
        self.main_code.push(expr_code);
        Ok(())
    }

    /// Generate within statement
    fn generate_within_statement(&mut self, within_stmt: &WithinStatement) -> Result<()> {
        let target_code = self.generate_expression_code(&within_stmt.target)?;
        
        self.main_code.push(format!("# Within block for {}", target_code));
        self.main_code.push(format!("for _item in {}:", target_code));
        
        // Process within contents
        for content in &within_stmt.body.contents {
            match content {
                WithinContent::FuzzyEvaluate(fuzzy_block) => {
                    self.main_code.push(format!("    # Fuzzy evaluate: {}", fuzzy_block.name));
                }
                WithinContent::Biomechanical(bio_block) => {
                    self.main_code.push(format!("    # Biomechanical analysis: {}", bio_block.name));
                }
                WithinContent::PatternMatching(pattern_block) => {
                    let condition_code = self.generate_expression_code(&pattern_block.condition)?;
                    self.main_code.push(format!("    if {}:", condition_code));
                    self.main_code.push("        # Pattern matched".to_string());
                }
                _ => {
                    self.main_code.push("    # TODO: Implement within content".to_string());
                }
            }
        }
        
        Ok(())
    }

    /// Generate expression code
    fn generate_expression_code(&self, expr: &Expression) -> Result<String> {
        match expr {
            Expression::Identifier(name, _) => Ok(name.clone()),
            Expression::StringLiteral(s, _) => Ok(format!("\"{}\"", s)),
            Expression::NumberLiteral(n, _) => Ok(n.to_string()),
            Expression::BooleanLiteral(b, _) => Ok(b.to_string()),
            Expression::Range(start, end, _) => {
                let start_code = self.generate_expression_code(start)?;
                let end_code = self.generate_expression_code(end)?;
                Ok(format!("range({}, {})", start_code, end_code))
            }
            Expression::List(items, _) => {
                let item_codes: Result<Vec<String>, _> = items.iter()
                    .map(|item| self.generate_expression_code(item))
                    .collect();
                Ok(format!("[{}]", item_codes?.join(", ")))
            }
            Expression::Dictionary(items, _) => {
                let mut dict_items = Vec::new();
                for (key, value) in items {
                    let value_code = self.generate_expression_code(value)?;
                    dict_items.push(format!("'{}': {}", key, value_code));
                }
                Ok(format!("{{{}}}", dict_items.join(", ")))
            }
            Expression::FunctionCall(call) => {
                let arg_codes: Result<Vec<String>, _> = call.arguments.iter()
                    .map(|arg| self.generate_expression_code(arg))
                    .collect();
                Ok(format!("{}({})", call.function, arg_codes?.join(", ")))
            }
            Expression::MemberAccess(access) => {
                let object_code = self.generate_expression_code(&access.object)?;
                Ok(format!("{}.{}", object_code, access.member))
            }
            Expression::BinaryOp(op) => {
                let left_code = self.generate_expression_code(&op.left)?;
                let right_code = self.generate_expression_code(&op.right)?;
                let op_str = match op.operator {
                    BinaryOperator::Add => "+",
                    BinaryOperator::Sub => "-",
                    BinaryOperator::Mul => "*",
                    BinaryOperator::Div => "/",
                    BinaryOperator::Mod => "%",
                    BinaryOperator::Pow => "**",
                    BinaryOperator::Eq => "==",
                    BinaryOperator::Ne => "!=",
                    BinaryOperator::Lt => "<",
                    BinaryOperator::Gt => ">",
                    BinaryOperator::Le => "<=",
                    BinaryOperator::Ge => ">=",
                    BinaryOperator::And => "and",
                    BinaryOperator::Or => "or",
                    BinaryOperator::In => "in",
                };
                Ok(format!("({} {} {})", left_code, op_str, right_code))
            }
            Expression::UnaryOp(op) => {
                let operand_code = self.generate_expression_code(&op.operand)?;
                let op_str = match op.operator {
                    UnaryOperator::Not => "not",
                    UnaryOperator::Minus => "-",
                };
                Ok(format!("({} {})", op_str, operand_code))
            }
        }
    }

    /// Get current indentation
    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_codegen() {
        let mut codegen = PythonCodeGenerator::new();
        
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
        
        let result = codegen.generate_variable_declaration(&var_decl);
        assert!(result.is_ok());
    }
} 