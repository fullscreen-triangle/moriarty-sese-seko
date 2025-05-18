#!/usr/bin/env python3
"""
Visualization Script for Moriarty Results

This script generates visualizations and charts from biomechanical data 
in the public/results folder. It can create various plots from existing
analysis data.
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from pathlib import Path


def setup_style():
    """Set up matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    # Custom color palette
    palette = sns.color_palette("mako_r", 6)
    sns.set_palette(palette)
    

def load_json_data(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_csv_data(filepath):
    """Load CSV data from a file."""
    return pd.read_csv(filepath)


def find_result_files(data_dir, pattern="*.json"):
    """Find all result files matching the pattern in the data directory."""
    return glob.glob(os.path.join(data_dir, pattern))


def plot_joint_angles(data, output_path=None):
    """
    Plot joint angles over time.
    
    Args:
        data: Dictionary or DataFrame containing joint angle data
        output_path: Path to save the plot
    """
    if isinstance(data, dict):
        # Convert dict to DataFrame if necessary
        df = pd.DataFrame(data)
    else:
        df = data
    
    plt.figure(figsize=(14, 8))
    
    # Plot joint angles if they exist in the data
    angle_columns = [col for col in df.columns if 'angle' in col.lower()]
    
    if not angle_columns:
        print("No joint angle data found")
        return
    
    for col in angle_columns:
        plt.plot(df.index, df[col], label=col.replace('_angle', '').title())
    
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Time')
    plt.legend(loc='best')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved joint angle plot to {output_path}")
    else:
        plt.show()


def plot_velocity_profile(data, output_path=None):
    """
    Plot velocity profile over time or distance.
    
    Args:
        data: Dictionary or DataFrame containing velocity data
        output_path: Path to save the plot
    """
    if isinstance(data, dict):
        # Convert dict to DataFrame if necessary
        df = pd.DataFrame(data)
    else:
        df = data
    
    plt.figure(figsize=(14, 8))
    
    # Look for velocity data
    vel_columns = [col for col in df.columns if 'velocity' in col.lower() or 'speed' in col.lower()]
    
    if not vel_columns:
        print("No velocity data found")
        return
    
    for col in vel_columns:
        label = col.replace('_velocity', '').replace('_speed', '').title()
        plt.plot(df.index, df[col], label=label)
    
    plt.xlabel('Frame')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profile')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved velocity profile to {output_path}")
    else:
        plt.show()


def plot_stride_parameters(data, output_path=None):
    """
    Plot stride parameters (length, frequency, etc.).
    
    Args:
        data: Dictionary or DataFrame containing stride data
        output_path: Path to save the plot
    """
    if isinstance(data, dict):
        # Convert stride events to DataFrame if necessary
        if 'stride_events' in data:
            strides = pd.DataFrame(data['stride_events'])
        elif 'strides' in data:
            strides = pd.DataFrame(data['strides'])
        else:
            strides = pd.DataFrame(data)
    else:
        strides = data
    
    if strides.empty:
        print("No stride data found")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot stride length if available
    if 'stride_length' in strides.columns:
        ax1.plot(strides.index, strides['stride_length'], 'o-', label='Stride Length')
        ax1.set_ylabel('Stride Length (m)')
        ax1.set_title('Stride Length Over Time')
        ax1.grid(True)
        ax1.legend()
    
    # Plot stride frequency/cadence if available
    stride_freq_cols = [col for col in strides.columns if any(x in col.lower() for x in ['frequency', 'cadence'])]
    if stride_freq_cols:
        for col in stride_freq_cols:
            ax2.plot(strides.index, strides[col], 's-', label=col.replace('_', ' ').title())
        ax2.set_ylabel('Stride Rate (steps/min)')
        ax2.set_title('Stride Rate Over Time')
        ax2.grid(True)
        ax2.legend()
    
    plt.xlabel('Stride Number')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved stride parameters plot to {output_path}")
    else:
        plt.show()


def plot_ground_reaction_forces(data, output_path=None):
    """
    Plot estimated ground reaction forces.
    
    Args:
        data: Dictionary or DataFrame containing GRF data
        output_path: Path to save the plot
    """
    if isinstance(data, dict):
        if 'grf' in data:
            grf_data = pd.DataFrame(data['grf'])
        else:
            # Try to find GRF-related keys
            grf_keys = [k for k in data.keys() if 'force' in k.lower() or 'grf' in k.lower()]
            if not grf_keys:
                print("No GRF data found")
                return
            grf_data = pd.DataFrame({k: data[k] for k in grf_keys})
    else:
        grf_data = data
    
    # Check for common GRF components
    components = []
    for component in ['vertical', 'anterior_posterior', 'medial_lateral', 'resultant']:
        cols = [col for col in grf_data.columns if component in col.lower()]
        if cols:
            components.extend(cols)
    
    if not components:
        # Try common GRF naming patterns
        components = [col for col in grf_data.columns if any(x in col.lower() for x in ['fz', 'fy', 'fx', 'grf'])]
    
    if not components:
        print("No GRF components identified in data")
        return
    
    plt.figure(figsize=(14, 8))
    
    for component in components:
        plt.plot(grf_data.index, grf_data[component], label=component.replace('_', ' ').title())
    
    plt.xlabel('Frame')
    plt.ylabel('Force (N)')
    plt.title('Ground Reaction Forces')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved GRF plot to {output_path}")
    else:
        plt.show()


def plot_power_metrics(data, output_path=None):
    """
    Plot power and work metrics for joints.
    
    Args:
        data: Dictionary or DataFrame containing power/work data
        output_path: Path to save the plot
    """
    if isinstance(data, dict):
        if 'joint_power' in data:
            power_data = pd.DataFrame(data['joint_power'])
        else:
            # Try to find power-related keys
            power_keys = [k for k in data.keys() if 'power' in k.lower() or 'work' in k.lower()]
            if not power_keys:
                print("No power data found")
                return
            power_data = pd.DataFrame({k: data[k] for k in power_keys})
    else:
        power_data = data
        
    # Find joint power columns
    joints = ['ankle', 'knee', 'hip']
    power_cols = []
    
    for joint in joints:
        cols = [col for col in power_data.columns if joint in col.lower() and 'power' in col.lower()]
        if cols:
            power_cols.extend(cols)
    
    if not power_cols:
        print("No joint power data found")
        return
    
    plt.figure(figsize=(14, 8))
    
    for col in power_cols:
        plt.plot(power_data.index, power_data[col], label=col.replace('_', ' ').title())
    
    plt.xlabel('Frame')
    plt.ylabel('Power (W/kg)')
    plt.title('Joint Power')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved power metrics plot to {output_path}")
    else:
        plt.show()


def create_performance_dashboard(data_dict, output_path=None):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        data_dict: Dictionary of DataFrames with different metrics
        output_path: Path to save the dashboard
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Plot joint angles
    if 'kinematics' in data_dict:
        ax1 = fig.add_subplot(gs[0, 0])
        angle_cols = [col for col in data_dict['kinematics'].columns if 'angle' in col.lower()]
        for col in angle_cols[:3]:  # Limit to 3 major joints for clarity
            ax1.plot(data_dict['kinematics'].index, data_dict['kinematics'][col], 
                     label=col.replace('_angle', '').title())
        ax1.set_title('Joint Angles')
        ax1.set_ylabel('Angle (degrees)')
        ax1.legend()
        ax1.grid(True)
    
    # Plot velocity profile
    if 'kinematics' in data_dict:
        ax2 = fig.add_subplot(gs[0, 1])
        vel_cols = [col for col in data_dict['kinematics'].columns 
                   if 'velocity' in col.lower() or 'speed' in col.lower()]
        for col in vel_cols[:2]:  # Limit to COM and maybe one other
            ax2.plot(data_dict['kinematics'].index, data_dict['kinematics'][col], 
                     label=col.replace('_', ' ').title())
        ax2.set_title('Velocity Profile')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True)
    
    # Plot stride parameters
    if 'strides' in data_dict:
        ax3 = fig.add_subplot(gs[1, 0])
        if 'stride_length' in data_dict['strides'].columns:
            ax3.plot(data_dict['strides'].index, data_dict['strides']['stride_length'], 'o-')
        ax3.set_title('Stride Length')
        ax3.set_ylabel('Length (m)')
        ax3.grid(True)
    
    # Plot GRF if available
    if 'grf' in data_dict:
        ax4 = fig.add_subplot(gs[1, 1])
        grf_cols = [col for col in data_dict['grf'].columns 
                    if any(x in col.lower() for x in ['vertical', 'resultant', 'fz'])]
        for col in grf_cols[:2]:
            ax4.plot(data_dict['grf'].index, data_dict['grf'][col], 
                     label=col.replace('_', ' ').title())
        ax4.set_title('Ground Reaction Forces')
        ax4.set_ylabel('Force (N)')
        ax4.legend()
        ax4.grid(True)
    
    # Plot power metrics if available
    if 'power' in data_dict:
        ax5 = fig.add_subplot(gs[2, 0])
        power_cols = [col for col in data_dict['power'].columns 
                      if 'power' in col.lower() and any(joint in col.lower() 
                                                        for joint in ['ankle', 'knee', 'hip'])]
        for col in power_cols:
            ax5.plot(data_dict['power'].index, data_dict['power'][col], 
                     label=col.replace('_', ' ').title())
        ax5.set_title('Joint Power')
        ax5.set_ylabel('Power (W/kg)')
        ax5.legend()
        ax5.grid(True)
    
    # Plot any other metrics (e.g., VO2, efficiency)
    if 'metrics' in data_dict:
        ax6 = fig.add_subplot(gs[2, 1])
        metric_cols = [col for col in data_dict['metrics'].columns 
                       if any(x in col.lower() for x in ['vo2', 'efficiency', 'economy'])]
        for col in metric_cols:
            ax6.plot(data_dict['metrics'].index, data_dict['metrics'][col], 
                     label=col.replace('_', ' ').title())
        ax6.set_title('Performance Metrics')
        ax6.set_ylabel('Value')
        ax6.legend()
        ax6.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved performance dashboard to {output_path}")
    else:
        plt.show()


def process_results_folder(results_dir, output_dir=None):
    """
    Process all results in the given directory and create visualizations.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save visualizations (defaults to results_dir/visualizations)
    """
    if not output_dir:
        output_dir = os.path.join(results_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON result files
    json_files = find_result_files(results_dir, "*.json")
    print(f"Found {len(json_files)} JSON result files")
    
    # Find all CSV result files
    csv_files = find_result_files(results_dir, "*.csv")
    print(f"Found {len(csv_files)} CSV result files")
    
    # Process each file
    for file_path in json_files:
        try:
            print(f"Processing {file_path}")
            filename = os.path.basename(file_path)
            basename = os.path.splitext(filename)[0]
            
            # Load data
            data = load_json_data(file_path)
            
            # Create directory for this result
            result_output_dir = os.path.join(output_dir, basename)
            os.makedirs(result_output_dir, exist_ok=True)
            
            # Generate plots
            plot_joint_angles(data, os.path.join(result_output_dir, 'joint_angles.png'))
            plot_velocity_profile(data, os.path.join(result_output_dir, 'velocity_profile.png'))
            plot_stride_parameters(data, os.path.join(result_output_dir, 'stride_parameters.png'))
            plot_ground_reaction_forces(data, os.path.join(result_output_dir, 'grf.png'))
            plot_power_metrics(data, os.path.join(result_output_dir, 'power_metrics.png'))
            
            # Try to create a dashboard
            try:
                # This assumes the data might be structured in a specific way
                data_dict = {}
                
                # Try to organize data into categories
                if isinstance(data, dict):
                    # Check for high-level categories
                    for category in ['kinematics', 'kinetics', 'strides', 'grf', 'power', 'metrics']:
                        if category in data:
                            data_dict[category] = pd.DataFrame(data[category])
                    
                    # If no categories found, infer from column names
                    if not data_dict:
                        df = pd.DataFrame(data)
                        data_dict['kinematics'] = df
                
                create_performance_dashboard(data_dict, os.path.join(result_output_dir, 'dashboard.png'))
            except Exception as e:
                print(f"Could not create dashboard: {e}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process CSV files
    for file_path in csv_files:
        try:
            print(f"Processing {file_path}")
            filename = os.path.basename(file_path)
            basename = os.path.splitext(filename)[0]
            
            # Load data
            df = load_csv_data(file_path)
            
            # Create directory for this result
            result_output_dir = os.path.join(output_dir, basename)
            os.makedirs(result_output_dir, exist_ok=True)
            
            # Generate plots based on column names
            if any('angle' in col.lower() for col in df.columns):
                plot_joint_angles(df, os.path.join(result_output_dir, 'joint_angles.png'))
            
            if any('velocity' in col.lower() or 'speed' in col.lower() for col in df.columns):
                plot_velocity_profile(df, os.path.join(result_output_dir, 'velocity_profile.png'))
            
            if any('stride' in col.lower() or 'cadence' in col.lower() for col in df.columns):
                plot_stride_parameters(df, os.path.join(result_output_dir, 'stride_parameters.png'))
            
            if any('force' in col.lower() or 'grf' in col.lower() for col in df.columns):
                plot_ground_reaction_forces(df, os.path.join(result_output_dir, 'grf.png'))
            
            if any('power' in col.lower() or 'work' in col.lower() for col in df.columns):
                plot_power_metrics(df, os.path.join(result_output_dir, 'power_metrics.png'))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Visualization generation complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from Moriarty results.')
    
    parser.add_argument('--results-dir', '-r', type=str, default='public/results',
                       help='Directory containing result files (default: public/results)')
    
    parser.add_argument('--output-dir', '-o', type=str, 
                       help='Directory to save visualizations (default: <results_dir>/visualizations)')
    
    parser.add_argument('--file', '-f', type=str,
                       help='Process a specific result file instead of the whole directory')
    
    parser.add_argument('--plot-type', '-p', type=str, choices=['joint_angles', 'velocity', 'stride', 'grf', 'power', 'dashboard', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Set plot style
    setup_style()
    
    if args.file:
        # Process single file
        file_path = args.file
        output_dir = args.output_dir or os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing single file: {file_path}")
        basename = os.path.splitext(os.path.basename(file_path))[0]
        
        if file_path.endswith('.json'):
            data = load_json_data(file_path)
        elif file_path.endswith('.csv'):
            data = load_csv_data(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return
        
        # Generate requested plot type
        if args.plot_type == 'joint_angles' or args.plot_type == 'all':
            plot_joint_angles(data, os.path.join(output_dir, f"{basename}_joint_angles.png"))
        
        if args.plot_type == 'velocity' or args.plot_type == 'all':
            plot_velocity_profile(data, os.path.join(output_dir, f"{basename}_velocity.png"))
        
        if args.plot_type == 'stride' or args.plot_type == 'all':
            plot_stride_parameters(data, os.path.join(output_dir, f"{basename}_stride.png"))
        
        if args.plot_type == 'grf' or args.plot_type == 'all':
            plot_ground_reaction_forces(data, os.path.join(output_dir, f"{basename}_grf.png"))
        
        if args.plot_type == 'power' or args.plot_type == 'all':
            plot_power_metrics(data, os.path.join(output_dir, f"{basename}_power.png"))
        
        if args.plot_type == 'dashboard' or args.plot_type == 'all':
            if isinstance(data, dict):
                data_dict = {}
                for category in ['kinematics', 'kinetics', 'strides', 'grf', 'power', 'metrics']:
                    if category in data:
                        data_dict[category] = pd.DataFrame(data[category])
                
                if not data_dict:  # If no categories found, use the whole dataset
                    data_dict = {'kinematics': pd.DataFrame(data)}
            else:
                data_dict = {'kinematics': data}
            
            create_performance_dashboard(data_dict, os.path.join(output_dir, f"{basename}_dashboard.png"))
    else:
        # Process entire directory
        process_results_folder(args.results_dir, args.output_dir)
    
    print("Visualization generation complete!")


if __name__ == "__main__":
    main() 