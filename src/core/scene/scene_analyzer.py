import numpy as np
from typing import Dict, List, Any, Optional
import logging
from .scene_detector import SceneDetector, SceneChange
from .analyzer import PoseAnalyzer
from .video_quality import VideoQualityAnalyzer


class SceneAnalyzer:
    """
    Comprehensive scene analysis that integrates multiple scene components.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the scene analyzer.
        
        Args:
            config: Configuration dictionary for various analyzers
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        if config is None:
            config = {
                'scene_detection': {
                    'hist_threshold': 0.5,
                    'flow_threshold': 0.7,
                    'edge_threshold': 0.6,
                    'focus_threshold': 1000
                },
                'output': {
                    'plots_directory': 'scene_analysis_plots'
                }
            }
        
        self.config = config
        
        # Initialize sub-analyzers
        self.scene_detector = SceneDetector(config)
        self.pose_analyzer = PoseAnalyzer()
        self.quality_analyzer = VideoQualityAnalyzer(config)
        
        # Analysis state
        self.scene_changes = []
        self.quality_metrics = []
        self.pose_analyses = []
    
    def analyze(self, scene_features: List[Dict]) -> Dict[str, Any]:
        """
        Analyze scene features and return comprehensive scene analysis.
        
        Args:
            scene_features: List of scene feature dictionaries from frame processing
            
        Returns:
            Dictionary containing comprehensive scene analysis results
        """
        if not scene_features:
            return {
                'scene_changes': [],
                'quality_summary': {},
                'pose_summary': {},
                'overall_quality': 'unknown',
                'scene_complexity': 0.0
            }
        
        try:
            # Extract and analyze different aspects
            quality_analysis = self._analyze_quality(scene_features)
            pose_analysis = self._analyze_poses(scene_features)
            scene_transitions = self._analyze_scene_transitions(scene_features)
            scene_complexity = self._calculate_scene_complexity(scene_features)
            
            # Generate overall assessment
            overall_quality = self._assess_overall_quality(quality_analysis, pose_analysis)
            
            return {
                'scene_changes': scene_transitions,
                'quality_summary': quality_analysis,
                'pose_summary': pose_analysis,
                'overall_quality': overall_quality,
                'scene_complexity': scene_complexity,
                'frame_count': len(scene_features),
                'analysis_metadata': {
                    'config': self.config,
                    'analyzers_used': ['scene_detector', 'pose_analyzer', 'quality_analyzer']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in scene analysis: {e}")
            return {
                'error': str(e),
                'scene_changes': [],
                'quality_summary': {},
                'pose_summary': {},
                'overall_quality': 'error',
                'scene_complexity': 0.0
            }
    
    def _analyze_quality(self, scene_features: List[Dict]) -> Dict[str, Any]:
        """Analyze video quality from scene features."""
        quality_metrics = []
        
        for frame_idx, features in enumerate(scene_features):
            if 'frame' in features and features['frame'] is not None:
                metrics = self.quality_analyzer.compute_frame_metrics(features['frame'])
                metrics['frame_idx'] = frame_idx
                quality_metrics.append(metrics)
                self.quality_analyzer.update_metrics_history(frame_idx, metrics)
        
        if not quality_metrics:
            return {'error': 'No valid frames for quality analysis'}
        
        # Calculate summary statistics
        brightness_values = [m.get('brightness', 0) for m in quality_metrics]
        contrast_values = [m.get('contrast', 0) for m in quality_metrics]
        sharpness_values = [m.get('sharpness', 0) for m in quality_metrics]
        noise_values = [m.get('noise', 0) for m in quality_metrics]
        saturation_values = [m.get('saturation', 0) for m in quality_metrics]
        
        return {
            'frame_count': len(quality_metrics),
            'brightness': {
                'mean': np.mean(brightness_values),
                'std': np.std(brightness_values),
                'min': np.min(brightness_values),
                'max': np.max(brightness_values)
            },
            'contrast': {
                'mean': np.mean(contrast_values),
                'std': np.std(contrast_values),
                'min': np.min(contrast_values),
                'max': np.max(contrast_values)
            },
            'sharpness': {
                'mean': np.mean(sharpness_values),
                'std': np.std(sharpness_values),
                'min': np.min(sharpness_values),
                'max': np.max(sharpness_values)
            },
            'noise': {
                'mean': np.mean(noise_values),
                'std': np.std(noise_values),
                'min': np.min(noise_values),
                'max': np.max(noise_values)
            },
            'saturation': {
                'mean': np.mean(saturation_values),
                'std': np.std(saturation_values),
                'min': np.min(saturation_values),
                'max': np.max(saturation_values)
            }
        }
    
    def _analyze_poses(self, scene_features: List[Dict]) -> Dict[str, Any]:
        """Analyze pose data from scene features."""
        pose_analyses = []
        valid_poses = 0
        
        for features in scene_features:
            if 'poses' in features:
                poses = features['poses']
                if poses:
                    for pose in poses:
                        if pose and 'pose_landmarks' in pose:
                            analysis = self.pose_analyzer.analyze_pose(pose['pose_landmarks'])
                            if analysis:
                                pose_analyses.append(analysis)
                                valid_poses += 1
        
        if not pose_analyses:
            return {'error': 'No valid poses found for analysis'}
        
        # Calculate pose summary statistics
        joint_angles_summary = self._summarize_joint_angles(pose_analyses)
        ground_contact_summary = self._summarize_ground_contact(pose_analyses)
        body_metrics_summary = self._summarize_body_metrics(pose_analyses)
        
        return {
            'total_poses': valid_poses,
            'frames_with_poses': len([f for f in scene_features if f.get('poses')]),
            'joint_angles': joint_angles_summary,
            'ground_contact': ground_contact_summary,
            'body_metrics': body_metrics_summary
        }
    
    def _analyze_scene_transitions(self, scene_features: List[Dict]) -> List[Dict]:
        """Analyze scene transitions and changes."""
        if len(scene_features) < 2:
            return []
        
        # Create a generator of frames for scene detection
        def frame_generator():
            for features in scene_features:
                if 'frame' in features and features['frame'] is not None:
                    yield features['frame']
        
        # Detect scene changes
        scene_changes = self.scene_detector.detect_scenes(frame_generator())
        
        # Convert to serializable format
        transitions = []
        for change in scene_changes:
            transitions.append({
                'frame_idx': change.frame_idx,
                'change_type': change.change_type.value,
                'confidence': change.confidence,
                'metrics': change.metrics
            })
        
        return transitions
    
    def _calculate_scene_complexity(self, scene_features: List[Dict]) -> float:
        """Calculate overall scene complexity score."""
        try:
            complexity_factors = []
            
            # Factor 1: Number of people in scene
            people_counts = []
            for features in scene_features:
                poses = features.get('poses', [])
                people_counts.append(len(poses) if poses else 0)
            
            avg_people = np.mean(people_counts) if people_counts else 0
            people_complexity = min(1.0, avg_people / 5.0)  # Normalize to max 5 people
            complexity_factors.append(people_complexity)
            
            # Factor 2: Scene change frequency
            transitions = self._analyze_scene_transitions(scene_features)
            change_frequency = len(transitions) / len(scene_features) if scene_features else 0
            change_complexity = min(1.0, change_frequency * 10)  # Normalize
            complexity_factors.append(change_complexity)
            
            # Factor 3: Quality variation (higher variation = more complex)
            quality_analysis = self._analyze_quality(scene_features)
            if 'brightness' in quality_analysis and 'std' in quality_analysis['brightness']:
                brightness_variation = quality_analysis['brightness']['std'] / 255.0
                quality_complexity = min(1.0, brightness_variation * 2)
                complexity_factors.append(quality_complexity)
            
            # Combine factors
            overall_complexity = np.mean(complexity_factors) if complexity_factors else 0.0
            return float(overall_complexity)
            
        except Exception as e:
            self.logger.error(f"Error calculating scene complexity: {e}")
            return 0.0
    
    def _assess_overall_quality(self, quality_analysis: Dict, pose_analysis: Dict) -> str:
        """Assess overall scene quality based on multiple factors."""
        try:
            quality_score = 0.0
            factors = 0
            
            # Quality factors
            if 'brightness' in quality_analysis:
                brightness_mean = quality_analysis['brightness'].get('mean', 0)
                # Good brightness range is around 100-150 (out of 255)
                brightness_score = 1.0 - abs(brightness_mean - 125) / 125
                quality_score += max(0, brightness_score)
                factors += 1
            
            if 'sharpness' in quality_analysis:
                sharpness_mean = quality_analysis['sharpness'].get('mean', 0)
                # Higher sharpness is better (threshold around 100)
                sharpness_score = min(1.0, sharpness_mean / 100)
                quality_score += sharpness_score
                factors += 1
            
            if 'noise' in quality_analysis:
                noise_mean = quality_analysis['noise'].get('mean', 0)
                # Lower noise is better (threshold around 10)
                noise_score = max(0, 1.0 - noise_mean / 10)
                quality_score += noise_score
                factors += 1
            
            # Pose factors
            if 'total_poses' in pose_analysis:
                pose_count = pose_analysis['total_poses']
                # Having poses is good for analysis
                pose_score = 1.0 if pose_count > 0 else 0.0
                quality_score += pose_score
                factors += 1
            
            if factors > 0:
                final_score = quality_score / factors
                
                if final_score >= 0.8:
                    return 'excellent'
                elif final_score >= 0.6:
                    return 'good'
                elif final_score >= 0.4:
                    return 'fair'
                else:
                    return 'poor'
            
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Error assessing overall quality: {e}")
            return 'error'
    
    def _summarize_joint_angles(self, pose_analyses: List[Dict]) -> Dict[str, Any]:
        """Summarize joint angle statistics."""
        try:
            angles_data = {
                'knee_angle': [],
                'hip_angle': [],
                'ankle_angle': [],
                'shoulder_angle': [],
                'elbow_angle': []
            }
            
            for analysis in pose_analyses:
                joint_angles = analysis.get('joint_angles')
                if joint_angles:
                    for angle_name in angles_data.keys():
                        if hasattr(joint_angles, angle_name):
                            angle_value = getattr(joint_angles, angle_name)
                            if angle_value and not np.isnan(angle_value):
                                angles_data[angle_name].append(angle_value)
            
            summary = {}
            for angle_name, values in angles_data.items():
                if values:
                    summary[angle_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
                else:
                    summary[angle_name] = {'count': 0}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing joint angles: {e}")
            return {}
    
    def _summarize_ground_contact(self, pose_analyses: List[Dict]) -> Dict[str, Any]:
        """Summarize ground contact statistics."""
        try:
            left_foot_contacts = []
            right_foot_contacts = []
            
            for analysis in pose_analyses:
                ground_contact = analysis.get('ground_contact')
                if ground_contact:
                    left_foot_contacts.append(ground_contact.get('left_foot_contact', False))
                    right_foot_contacts.append(ground_contact.get('right_foot_contact', False))
            
            return {
                'left_foot_contact_ratio': float(np.mean(left_foot_contacts)) if left_foot_contacts else 0.0,
                'right_foot_contact_ratio': float(np.mean(right_foot_contacts)) if right_foot_contacts else 0.0,
                'total_frames': len(left_foot_contacts)
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing ground contact: {e}")
            return {}
    
    def _summarize_body_metrics(self, pose_analyses: List[Dict]) -> Dict[str, Any]:
        """Summarize body metrics statistics."""
        try:
            heights = []
            stance_widths = []
            
            for analysis in pose_analyses:
                body_metrics = analysis.get('body_metrics')
                if body_metrics:
                    height = body_metrics.get('height')
                    stance_width = body_metrics.get('stance_width')
                    
                    if height and not np.isnan(height):
                        heights.append(height)
                    if stance_width and not np.isnan(stance_width):
                        stance_widths.append(stance_width)
            
            summary = {}
            
            if heights:
                summary['height'] = {
                    'mean': float(np.mean(heights)),
                    'std': float(np.std(heights)),
                    'min': float(np.min(heights)),
                    'max': float(np.max(heights))
                }
            
            if stance_widths:
                summary['stance_width'] = {
                    'mean': float(np.mean(stance_widths)),
                    'std': float(np.std(stance_widths)),
                    'min': float(np.min(stance_widths)),
                    'max': float(np.max(stance_widths))
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing body metrics: {e}")
            return {}
    
    def reset(self):
        """Reset analysis state for new video processing."""
        self.scene_changes = []
        self.quality_metrics = []
        self.pose_analyses = []
        self.scene_detector.reset_metrics()
        self.quality_analyzer.metrics_history = [] 