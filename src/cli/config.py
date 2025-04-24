#!/usr/bin/env python3
"""
Configuration module for the Moriarty CLI.

This module handles loading, validating, and generating YAML configuration files
for the Moriarty pipeline CLI.
"""

import os
import re
import yaml
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure module logger
logger = logging.getLogger("moriarty-cli")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Return empty dict if the file is empty
    if config is None:
        return {}
        
    return config

def save_config(config: Dict[Any, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Raises:
        IOError: If the file cannot be written
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        raise IOError(f"Failed to save configuration: {str(e)}")

def merge_cli_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge command line arguments into configuration dictionary.
    CLI arguments take precedence over configuration file values.
    
    Args:
        config: Base configuration dictionary
        args: Command line arguments dictionary
        
    Returns:
        Merged configuration dictionary
    """
    # Create a deep copy of the config to avoid modifying the original
    merged_config = dict(config)
    
    # Map CLI args to configuration structure
    if args.get('command') == 'analyze':
        # Analyze command mappings
        if args.get('video_path'):
            merged_config.setdefault('input', {})['video_path'] = args['video_path']
        
        if args.get('output_dir'):
            merged_config.setdefault('output', {})['directory'] = args['output_dir']
            
        if args.get('save_visualization'):
            merged_config.setdefault('visualization', {})['enabled'] = True
            
        if args.get('save_data'):
            merged_config.setdefault('output', {})['save_raw_data'] = True
            
        if args.get('model'):
            merged_config.setdefault('models', {})['analysis_model'] = args['model']
    
    elif args.get('command') == 'batch':
        # Batch command mappings
        if args.get('input_dir'):
            merged_config.setdefault('batch', {})['input_directory'] = args['input_dir']
            
        if args.get('output_dir'):
            merged_config.setdefault('batch', {})['output_directory'] = args['output_dir']
            
        if args.get('pattern'):
            merged_config.setdefault('batch', {})['file_pattern'] = args['pattern']
            
        if args.get('parallel'):
            merged_config.setdefault('batch', {})['parallel_jobs'] = args['parallel']
            
        if args.get('save_visualization'):
            merged_config.setdefault('visualization', {})['enabled'] = True
            
        if args.get('save_data'):
            merged_config.setdefault('output', {})['save_raw_data'] = True
    
    elif args.get('command') == 'distill':
        # Distill command mappings
        if args.get('model'):
            merged_config.setdefault('knowledge_distillation', {})['base_model'] = args['model']
            
        if args.get('data'):
            merged_config.setdefault('knowledge_distillation', {})['data_directory'] = args['data']
            
        if args.get('output_dir'):
            merged_config.setdefault('knowledge_distillation', {})['output_directory'] = args['output_dir']
            
        if args.get('epochs'):
            merged_config.setdefault('knowledge_distillation', {})['epochs'] = args['epochs']
            
        if args.get('batch_size'):
            merged_config.setdefault('knowledge_distillation', {})['batch_size'] = args['batch_size']
            
        if args.get('examples'):
            merged_config.setdefault('knowledge_distillation', {})['num_examples'] = args['examples']
            
    elif args.get('command') == 'report':
        # Report command mappings
        if args.get('input_dir'):
            merged_config.setdefault('report', {})['input_directory'] = args['input_dir']
            
        if args.get('output_file'):
            merged_config.setdefault('report', {})['output_file'] = args['output_file']
            
        if args.get('format'):
            merged_config.setdefault('report', {})['format'] = args['format']
            
    elif args.get('command') == 'visualize':
        # Visualize command mappings
        if args.get('input_file'):
            merged_config.setdefault('visualization', {})['input_file'] = args['input_file']
            
        if args.get('output_file'):
            merged_config.setdefault('visualization', {})['output_file'] = args['output_file']
            
        if args.get('type'):
            merged_config.setdefault('visualization', {})['type'] = args['type']
    
    # Global options
    if args.get('log_level'):
        merged_config.setdefault('logging', {})['level'] = args['log_level']
        
    if args.get('log_file'):
        merged_config.setdefault('logging', {})['file'] = args['log_file']
    
    return merged_config

def set_config_value(config: Dict[Any, Any], path: str, value: Any) -> None:
    """
    Set a configuration value using a dot-notation path.
    
    Args:
        config: Configuration dictionary to modify
        path: Dot-notation path (e.g., 'section.subsection.key')
        value: Value to set
    """
    if not path:
        return
        
    parts = path.split('.')
    current = config
    
    # Navigate to the deepest dict
    for i, part in enumerate(parts[:-1]):
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    
    # Set the value
    current[parts[-1]] = value

def get_config_value(config: Dict[Any, Any], path: str, default: Any = None) -> Any:
    """
    Get a configuration value using a dot-notation path.
    
    Args:
        config: Configuration dictionary
        path: Dot-notation path (e.g., 'section.subsection.key')
        default: Default value to return if the path does not exist
        
    Returns:
        Value at the specified path, or default if not found
    """
    if not path:
        return default
        
    parts = path.split('.')
    current = config
    
    # Navigate through the dict
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    
    return current

def generate_default_config(output_path: str) -> None:
    """
    Generate a default configuration file.
    
    Args:
        output_path: Path to save the configuration file
    """
    # Create default configuration
    default_config = {
        # Input configuration
        'input': {
            'video_path': '',
            'data_directory': 'data/input',
        },
        
        # Output configuration
        'output': {
            'directory': 'output',
            'save_raw_data': True,
            'file_format': 'json',
        },
        
        # Visualization settings
        'visualization': {
            'enabled': True,
            'type': 'overlay',
            'show_skeleton': True,
            'show_metrics': True,
            'color_scheme': 'default',
            'output_fps': 30,
        },
        
        # Model settings
        'models': {
            'pose_model': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
            'track_model': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'analysis_model': 'biomechanics/dynamics_v2',
        },
        
        # Processing settings
        'processing': {
            'frame_sample_rate': 1,  # Process every nth frame
            'detection_threshold': 0.7,
            'min_track_length': 15,  # Minimum frames to consider a valid track
            'smoothing_window': 5,
        },
        
        # Batch processing settings
        'batch': {
            'input_directory': 'data/videos',
            'output_directory': 'output',
            'file_pattern': '*.mp4',
            'parallel_jobs': 1,
            'summary_report': True,
        },
        
        # Knowledge distillation settings
        'knowledge_distillation': {
            'base_model': 'google/flan-t5-base',
            'data_directory': 'data/biomechanics',
            'output_directory': 'models/distilled',
            'epochs': 3,
            'batch_size': 8,
            'num_examples': 1000,
            'learning_rate': 5e-5,
            'max_length': 512,
            'use_fp16': False,
        },
        
        # Report generation settings
        'report': {
            'template': 'default',
            'include_charts': True,
            'include_thumbnails': True,
            'format': 'pdf',
        },
        
        # Logging settings
        'logging': {
            'level': 'info',
            'file': '',
            'log_directory': 'logs',
        },
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write configuration to file
    with open(output_path, 'w') as file:
        yaml.safe_dump(default_config, file, default_flow_style=False, sort_keys=False)

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation issues (empty list if valid)
    """
    issues = []
    
    # Check if required sections exist
    required_sections = []
    if config.get('command') == 'analyze':
        required_sections = ['input']
    elif config.get('command') == 'batch':
        required_sections = ['batch']
    elif config.get('command') == 'distill':
        required_sections = ['knowledge_distillation']
    
    for section in required_sections:
        if section not in config:
            issues.append(f"Required section '{section}' is missing")
    
    # Check visualization settings
    vis_section = config.get('visualization', {})
    if vis_section.get('enabled', False):
        if vis_section.get('type') and vis_section.get('type') not in ['overlay', 'side-by-side', 'metrics']:
            issues.append(f"Invalid visualization type: {vis_section.get('type')}")
    
    # Check knowledge distillation settings
    kd_section = config.get('knowledge_distillation', {})
    if kd_section:
        if 'base_model' in kd_section and not isinstance(kd_section['base_model'], str):
            issues.append("base_model in knowledge_distillation must be a string")
        
        if 'epochs' in kd_section and not isinstance(kd_section['epochs'], int):
            issues.append("epochs in knowledge_distillation must be an integer")
            
        if 'batch_size' in kd_section and not isinstance(kd_section['batch_size'], int):
            issues.append("batch_size in knowledge_distillation must be an integer")
            
        if 'num_examples' in kd_section and not isinstance(kd_section['num_examples'], int):
            issues.append("num_examples in knowledge_distillation must be an integer")
    
    # Check batch processing settings
    batch_section = config.get('batch', {})
    if batch_section:
        if 'parallel_jobs' in batch_section and not isinstance(batch_section['parallel_jobs'], int):
            issues.append("parallel_jobs in batch must be an integer")
            
        if 'file_pattern' in batch_section and not isinstance(batch_section['file_pattern'], str):
            issues.append("file_pattern in batch must be a string")
    
    return issues 