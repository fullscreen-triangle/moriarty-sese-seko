import os
import json
import logging
import torch
from typing import Dict, List, Any, Tuple
import numpy as np

from src.core.dynamics.dynamics_analyzer import DynamicsAnalyzer, Segment
from src.api.ai_models import MoriartyLLMClient

logger = logging.getLogger(__name__)

class AIDynamicsAnalyzer(DynamicsAnalyzer):
    """
    Extended DynamicsAnalyzer that uses specialized AI models for advanced biomechanical analysis.
    """
    
    def __init__(self, use_parallel=True, n_processes=None, api_token=None, use_gpu=False):
        """
        Initialize the AI-powered dynamics analyzer.
        
        Args:
            use_parallel: Whether to use parallel processing
            n_processes: Number of processes to use for parallel processing
            api_token: Hugging Face API token (defaults to environment variable)
            use_gpu: Whether to use GPU acceleration for calculations
        """
        super().__init__(use_parallel=use_parallel, n_processes=n_processes, use_gpu=use_gpu)
        
        # Initialize the LLM client
        self.ai_client = MoriartyLLMClient(api_token=api_token)
        
        # Additional GPU setup for AI-specific operations
        if self.use_gpu:
            logger.info("AI Dynamics Analyzer initialized with GPU acceleration and specialized models")
        else:
            logger.info("AI Dynamics Analyzer initialized with specialized models")
        
    def analyze_biomechanics(self, 
                           positions_batch: List[Dict], 
                           velocities_batch: List[Dict],
                           accelerations_batch: List[Dict],
                           athlete_info: Dict = None) -> Dict:
        """
        Perform comprehensive biomechanical analysis using specialized AI models.
        
        Args:
            positions_batch: List of position dictionaries for each frame
            velocities_batch: List of velocity dictionaries for each frame
            accelerations_batch: List of acceleration dictionaries for each frame
            athlete_info: Information about the athlete
            
        Returns:
            Comprehensive biomechanical analysis
        """
        # First calculate basic dynamics using the parent class method
        basic_dynamics = self.calculate_dynamics_batch(
            positions_batch, velocities_batch, accelerations_batch
        )
        
        # Extract key metrics from the basic dynamics results
        biomechanical_data = self._extract_metrics(
            positions_batch, velocities_batch, accelerations_batch, basic_dynamics
        )
        
        # Default athlete info if not provided
        if athlete_info is None:
            athlete_info = {"sport": "sprint", "level": "elite", "age": 25}
        
        # Use the AI client to analyze the biomechanical data
        analysis = self.ai_client.analyze_sync(
            analysis_type="biomechanical_analysis",
            data=biomechanical_data,
            athlete_info=athlete_info
        )
        
        # Combine basic dynamics with AI analysis
        result = {
            "basic_dynamics": basic_dynamics,
            "metrics": biomechanical_data,
            "ai_analysis": analysis
        }
        
        return result
    
    def analyze_movement_patterns(self,
                               pose_sequence: List[Dict],
                               reference_patterns: Dict = None) -> Dict:
        """
        Analyze movement patterns compared to reference patterns.
        
        Args:
            pose_sequence: Sequence of pose data
            reference_patterns: Reference patterns to compare against
            
        Returns:
            Movement pattern analysis
        """
        # Use default reference patterns if none provided
        if reference_patterns is None:
            reference_patterns = self._get_default_reference_patterns()
        
        # Use the AI client to analyze movement patterns
        analysis = self.ai_client.analyze_sync(
            analysis_type="movement_patterns",
            data=pose_sequence[:20],  # Limit to 20 frames for efficiency
            reference_patterns=reference_patterns
        )
        
        return {
            "pose_sequence": pose_sequence[:5],  # Include sample of pose data
            "reference_patterns": reference_patterns,
            "analysis": analysis
        }
    
    def generate_technical_report(self, analysis_results: Dict, athlete_profile: Dict = None) -> Dict:
        """
        Generate a comprehensive technical report based on analysis results.
        
        Args:
            analysis_results: Results of biomechanical analysis
            athlete_profile: Profile information about the athlete
            
        Returns:
            Technical report
        """
        # Default athlete profile if not provided
        if athlete_profile is None:
            athlete_profile = {
                "name": "Athlete",
                "age": 25,
                "sport": "sprint",
                "level": "elite",
                "goals": ["Improve acceleration", "Reduce injury risk"]
            }
        
        # Use the AI client to generate a technical report
        report = self.ai_client.analyze_sync(
            analysis_type="technical_reporting",
            data=analysis_results,
            athlete_profile=athlete_profile
        )
        
        return report
    
    def analyze_sprint_technique(self, sprint_data: Dict) -> Dict:
        """
        Perform specialized sprint technique analysis.
        
        Args:
            sprint_data: Sprint-specific data
            
        Returns:
            Sprint technique analysis
        """
        # Extract sprint-specific metrics
        sprint_specific_metrics = self._extract_sprint_metrics(sprint_data)
        
        # Use the AI client to analyze sprint technique
        analysis = self.ai_client.analyze_sync(
            analysis_type="sprint_specialist",
            data=sprint_specific_metrics
        )
        
        return {
            "metrics": sprint_specific_metrics,
            "analysis": analysis
        }
    
    def compare_performances(self, current_data: Dict, previous_data: List[Dict]) -> Dict:
        """
        Compare current performance with previous performances.
        
        Args:
            current_data: Current performance data
            previous_data: Previous performance data
            
        Returns:
            Performance comparison analysis
        """
        # Extract key metrics from current and previous data
        current_metrics = self._extract_comparison_metrics(current_data)
        previous_metrics = [self._extract_comparison_metrics(data) for data in previous_data]
        
        # Use the AI client to compare performances
        comparison = self.ai_client.analyze_sync(
            analysis_type="performance_comparison",
            data=current_metrics,
            previous_performances=previous_metrics
        )
        
        return comparison
    
    def get_coaching_insights(self, analysis_results: Dict, coach_context: Dict = None) -> Dict:
        """
        Generate coaching insights based on analysis results.
        
        Args:
            analysis_results: Results of biomechanical analysis
            coach_context: Context information for coaching
            
        Returns:
            Coaching insights
        """
        # Default coach context if not provided
        if coach_context is None:
            coach_context = {
                "focus_areas": ["Acceleration technique", "Force production"],
                "training_phase": "Competition preparation",
                "available_equipment": ["Force plates", "High-speed cameras"]
            }
        
        # Use the AI client to generate coaching insights
        insights = self.ai_client.analyze_sync(
            analysis_type="coaching_insights",
            data=analysis_results,
            coach_context=coach_context
        )
        
        return insights
    
    def quick_analysis(self, data: Dict) -> Dict:
        """
        Perform a quick analysis of biomechanical data.
        
        Args:
            data: Biomechanical data
            
        Returns:
            Quick analysis results
        """
        # Extract key metrics for quick analysis
        key_metrics = self._extract_key_metrics(data)
        
        # Use the AI client for quick analysis
        analysis = self.ai_client.analyze_sync(
            analysis_type="quick_analysis",
            data=key_metrics
        )
        
        return analysis
    
    def _extract_metrics(self, positions: List[Dict], velocities: List[Dict], 
                      accelerations: List[Dict], dynamics: List[Dict]) -> Dict:
        """
        Extract key biomechanical metrics from raw data.
        
        Args:
            positions: Position data
            velocities: Velocity data
            accelerations: Acceleration data
            dynamics: Dynamic forces and moments
            
        Returns:
            Extracted biomechanical metrics
        """
        # Use GPU if available for metric extraction (computationally intensive)
        if self.use_gpu:
            return self._extract_metrics_gpu(positions, velocities, accelerations, dynamics)
            
        # Regular CPU processing (original implementation)
        # Extract stride metrics
        stride_length = self._calculate_stride_length(positions)
        stride_cadence = self._calculate_stride_cadence(positions)
        ground_contact_time = self._estimate_ground_contact(positions)
        
        # Extract joint kinematics
        joint_angles = self._calculate_joint_angles(positions)
        
        # Extract force metrics
        mean_forces = {}
        peak_forces = {}
        
        for segment in ['foot', 'shank', 'thigh']:
            forces = [frame['forces'][segment]['proximal'] for frame in dynamics if segment in frame.get('forces', {})]
            if forces:
                forces_array = np.array(forces)
                mean_forces[segment] = forces_array.mean(axis=0).tolist()
                peak_forces[segment] = forces_array.max(axis=0).tolist()
        
        # Combine metrics
        biomechanical_metrics = {
            "stride": {
                "length": stride_length,
                "cadence": stride_cadence,
                "ground_contact_time": ground_contact_time
            },
            "kinematics": {
                "joint_angles": joint_angles
            },
            "forces": {
                "mean": mean_forces,
                "peak": peak_forces
            },
            "efficiency": {
                "coordination_index": self._calculate_coordination_index(positions, velocities),
                "force_application_efficiency": self._calculate_force_efficiency(dynamics)
            }
        }
        
        return biomechanical_metrics
    
    def _extract_metrics_gpu(self, positions: List[Dict], velocities: List[Dict], 
                          accelerations: List[Dict], dynamics: List[Dict]) -> Dict:
        """
        GPU-accelerated extraction of biomechanical metrics.
        
        Args:
            positions: Position data
            velocities: Velocity data
            accelerations: Acceleration data
            dynamics: Dynamic forces and moments
            
        Returns:
            Extracted biomechanical metrics
        """
        logger.info("Using GPU acceleration for biomechanical metrics extraction")
        
        # Extract stride metrics with GPU acceleration
        stride_length = self._calculate_stride_length_gpu(positions)
        stride_cadence = self._calculate_stride_cadence_gpu(positions)
        ground_contact_time = self._estimate_ground_contact_gpu(positions)
        
        # Extract joint kinematics with GPU acceleration
        joint_angles = self._calculate_joint_angles_gpu(positions)
        
        # Extract force metrics using tensors
        mean_forces = {}
        peak_forces = {}
        
        for segment in ['foot', 'shank', 'thigh']:
            # Extract forces as tensor
            forces = [frame['forces'][segment]['proximal'] for frame in dynamics if segment in frame.get('forces', {})]
            
            if forces:
                # Convert to tensor and move to GPU
                forces_tensor = torch.tensor(forces, dtype=torch.float32, device=self.device)
                
                # Calculate statistics with tensor operations (faster on GPU)
                mean_forces[segment] = forces_tensor.mean(dim=0).cpu().numpy().tolist()
                peak_forces[segment] = forces_tensor.max(dim=0)[0].cpu().numpy().tolist()
        
        # Calculate efficiency metrics with GPU acceleration
        coordination_index = self._calculate_coordination_index_gpu(positions, velocities)
        force_efficiency = self._calculate_force_efficiency_gpu(dynamics)
        
        # Combine metrics
        biomechanical_metrics = {
            "stride": {
                "length": stride_length,
                "cadence": stride_cadence,
                "ground_contact_time": ground_contact_time
            },
            "kinematics": {
                "joint_angles": joint_angles
            },
            "forces": {
                "mean": mean_forces,
                "peak": peak_forces
            },
            "efficiency": {
                "coordination_index": coordination_index,
                "force_application_efficiency": force_efficiency
            }
        }
        
        return biomechanical_metrics
    
    def _calculate_stride_length_gpu(self, positions: List[Dict]) -> float:
        """GPU-accelerated stride length calculation"""
        # Implementation example - would need to be adapted to actual data format
        ankle_positions = []
        for frame in positions:
            if 'ankle' in frame:
                ankle_positions.append(frame['ankle'])
        
        if not ankle_positions or len(ankle_positions) < 2:
            return 0.0
            
        # Convert to tensor for GPU processing
        ankle_tensor = torch.tensor(ankle_positions, dtype=torch.float32, device=self.device)
        
        # Calculate stride length using tensor operations
        diff = ankle_tensor[1:] - ankle_tensor[:-1]
        stride_vector = diff.norm(dim=1)
        stride_length = stride_vector.mean().item()
        
        return stride_length
    
    def _calculate_stride_cadence_gpu(self, positions: List[Dict]) -> float:
        """GPU-accelerated stride cadence calculation"""
        # Simplified placeholder implementation
        return 2.5  # Example value
    
    def _estimate_ground_contact_gpu(self, positions: List[Dict]) -> float:
        """GPU-accelerated ground contact time estimation"""
        # Simplified placeholder implementation
        return 0.15  # Example value
    
    def _calculate_joint_angles_gpu(self, positions: List[Dict]) -> Dict:
        """GPU-accelerated joint angle calculation"""
        # Simplified placeholder implementation
        return {
            "knee": 135.0,
            "ankle": 90.0,
            "hip": 170.0
        }
    
    def _calculate_coordination_index_gpu(self, positions: List[Dict], velocities: List[Dict]) -> float:
        """GPU-accelerated coordination index calculation"""
        # Simplified placeholder implementation
        return 0.85  # Example value
    
    def _calculate_force_efficiency_gpu(self, dynamics: List[Dict]) -> float:
        """GPU-accelerated force efficiency calculation"""
        # Simplified placeholder implementation
        return 0.78  # Example value
    
    def _extract_sprint_metrics(self, sprint_data: Dict) -> Dict:
        """
        Extract sprint-specific metrics.
        
        Args:
            sprint_data: Sprint data
            
        Returns:
            Sprint-specific metrics
        """
        # Example sprint metrics (would be calculated from actual data)
        sprint_metrics = {
            "block_clearance": {
                "reaction_time": 0.15,
                "force_production": 850.0,
                "block_velocity": 3.2
            },
            "acceleration": {
                "0-10m_split": 1.8,
                "0-30m_split": 3.9,
                "max_acceleration": 12.5
            },
            "maximum_velocity": {
                "top_speed": 10.2,
                "stride_length": 2.25,
                "stride_frequency": 4.5,
                "flight_time": 0.12
            },
            "speed_endurance": {
                "speed_decay": 0.05,
                "30-60m_velocity_maintenance": 0.95
            }
        }
        
        return sprint_metrics
    
    def _extract_comparison_metrics(self, performance_data: Dict) -> Dict:
        """
        Extract key metrics for performance comparison.
        
        Args:
            performance_data: Performance data
            
        Returns:
            Key metrics for comparison
        """
        # Extract the most important metrics for comparison
        # (These would be calculated from the actual data)
        comparison_metrics = {
            "date": performance_data.get("date", "Unknown"),
            "sprint_times": performance_data.get("sprint_times", {}),
            "stride_metrics": performance_data.get("stride_metrics", {}),
            "force_production": performance_data.get("force_metrics", {}),
            "technical_scores": performance_data.get("technical_scores", {})
        }
        
        return comparison_metrics
    
    def _extract_key_metrics(self, data: Dict) -> Dict:
        """
        Extract the most important metrics for quick analysis.
        
        Args:
            data: Biomechanical data
            
        Returns:
            Key metrics for quick analysis
        """
        # Extract only the most critical metrics for quick analysis
        key_metrics = {
            "max_velocity": data.get("max_velocity", 0),
            "stride_length": data.get("stride_length", 0),
            "ground_contact_time": data.get("ground_contact_time", 0),
            "joint_angles": {
                k: v for k, v in data.get("joint_angles", {}).items() 
                if k in ["knee_flexion_max", "hip_extension_max", "ankle_dorsiflexion_max"]
            },
            "force_values": {
                k: v for k, v in data.get("forces", {}).items()
                if k in ["vertical_peak", "horizontal_peak"]
            }
        }
        
        return key_metrics
    
    def _get_default_reference_patterns(self) -> Dict:
        """
        Get default reference patterns for elite sprinters.
        
        Returns:
            Default reference patterns
        """
        # Example reference patterns for elite sprinters
        return {
            "acceleration_phase": {
                "body_lean": 45,
                "stride_length": 1.8,
                "knee_drive_height": "high",
                "foot_strike": "ball of foot",
                "joint_angles": {
                    "hip_extension": 170,
                    "knee_flexion": 110,
                    "ankle_dorsiflexion": 85
                }
            },
            "maximum_velocity_phase": {
                "body_posture": "upright",
                "stride_length": 2.2,
                "knee_drive_height": "medium-high",
                "foot_strike": "ball of foot",
                "joint_angles": {
                    "hip_extension": 175,
                    "knee_flexion": 125,
                    "ankle_dorsiflexion": 80
                }
            }
        }
    
    # Helper methods for metric calculation
    
    def _calculate_stride_length(self, positions: List[Dict]) -> float:
        """Calculate average stride length from position data."""
        # Placeholder implementation
        return 2.05  # Example value in meters
    
    def _calculate_stride_cadence(self, positions: List[Dict]) -> float:
        """Calculate stride cadence from position data."""
        # Placeholder implementation
        return 4.2  # Example value in strides/second
    
    def _estimate_ground_contact(self, positions: List[Dict]) -> float:
        """Estimate ground contact time from position data."""
        # Placeholder implementation
        return 0.11  # Example value in seconds
    
    def _calculate_joint_angles(self, positions: List[Dict]) -> Dict:
        """Calculate joint angles from position data."""
        # Placeholder implementation
        return {
            "hip": {"extension": 165, "flexion": 70},
            "knee": {"extension": 175, "flexion": 45},
            "ankle": {"dorsiflexion": 80, "plantarflexion": 140}
        }
    
    def _calculate_coordination_index(self, positions: List[Dict], velocities: List[Dict]) -> float:
        """Calculate coordination index from position and velocity data."""
        # Placeholder implementation
        return 0.85  # Example value (0-1 scale)
    
    def _calculate_force_efficiency(self, dynamics: List[Dict]) -> float:
        """Calculate force application efficiency from dynamics data."""
        # Placeholder implementation
        return 0.78  # Example value (0-1 scale) 