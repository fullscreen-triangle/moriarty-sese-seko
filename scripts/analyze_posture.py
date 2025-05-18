#!/usr/bin/env python3
"""
Analyze Posture Data Script for Moriarty

This script processes biomechanical posture data from the public/results/posture
folder and generates comprehensive visualizations and analysis reports.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from scipy import stats


def setup_visualization_style():
    """Set up matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # Custom color palette designed for biomechanics visualization
    colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    sns.set_palette(sns.color_palette(colors))
    
    # Custom colormap for heatmaps
    cmap_colors = ['#2C3E50', '#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
    biomech_cmap = LinearSegmentedColormap.from_list("biomech", cmap_colors)
    plt.register_cmap(cmap=biomech_cmap)


def load_posture_data(file_path):
    """
    Load posture data from various file formats.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded data as dict or DataFrame
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def find_posture_data_files(data_dir, pattern="*.json"):
    """
    Find all posture data files matching the pattern in the data directory.
    
    Args:
        data_dir: Directory to search in
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    return glob.glob(os.path.join(data_dir, pattern))


def analyze_joint_angles(data, output_path=None):
    """
    Analyze joint angles from posture data.
    
    Args:
        data: Posture data (dict or DataFrame)
        output_path: Path to save the analysis results
        
    Returns:
        Joint angle analysis dict
    """
    if isinstance(data, dict):
        # Try to find joint angle data
        if 'joint_angles' in data:
            angles = data['joint_angles']
        else:
            # Look for keys that might contain angle data
            angle_keys = [k for k in data.keys() if 'angle' in k.lower()]
            if not angle_keys:
                print("No joint angle data found")
                return {}
                
            # Use the first angle-related key
            angles = data[angle_keys[0]]
            
        # Convert to DataFrame if not already
        if not isinstance(angles, pd.DataFrame):
            angles = pd.DataFrame(angles)
    else:
        angles = data
        
    # Identify joint angle columns
    angle_cols = [col for col in angles.columns if 'angle' in col.lower()]
    
    if not angle_cols:
        print("No joint angle columns found in data")
        return {}
        
    # Calculate statistics for each joint angle
    angle_stats = {}
    
    for col in angle_cols:
        joint_name = col.replace('_angle', '').title()
        values = angles[col].dropna().values
        
        if len(values) == 0:
            continue
            
        angle_stats[joint_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values))
        }
        
    # Create visualizations
    if angle_stats and output_path:
        # Bar chart of joint angle ranges
        plt.figure(figsize=(12, 8))
        joints = list(angle_stats.keys())
        ranges = [stats['range'] for stats in angle_stats.values()]
        
        bars = plt.bar(joints, ranges)
        
        # Color bars by range magnitude
        norm = plt.Normalize(min(ranges), max(ranges))
        colors = plt.cm.viridis(norm(ranges))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Joint Angle Range of Motion')
        plt.ylabel('Range (degrees)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
    return angle_stats


def analyze_posture_alignment(data, output_path=None):
    """
    Analyze posture alignment and deviations.
    
    Args:
        data: Posture data (dict or DataFrame)
        output_path: Path to save the analysis results
        
    Returns:
        Posture alignment analysis dict
    """
    if isinstance(data, dict):
        # Try to find posture alignment data
        if 'posture' in data:
            posture = data['posture']
        elif 'alignment' in data:
            posture = data['alignment']
        else:
            # Look for keys that might contain posture data
            posture_keys = [k for k in data.keys() if any(x in k.lower() for x in ['posture', 'alignment', 'deviation'])]
            if not posture_keys:
                print("No posture alignment data found")
                return {}
                
            # Use the first posture-related key
            posture = data[posture_keys[0]]
            
        # Convert to DataFrame if not already
        if not isinstance(posture, pd.DataFrame):
            posture = pd.DataFrame(posture)
    else:
        posture = data
        
    # Check for required columns
    required_cols = ['frame', 'alignment_score']
    has_required = all(col in posture.columns for col in required_cols)
    
    if not has_required:
        # Look for alternative column names
        alt_frame_cols = [col for col in posture.columns if any(x in col.lower() for x in ['frame', 'time', 'index'])]
        alt_score_cols = [col for col in posture.columns if any(x in col.lower() for x in ['score', 'alignment', 'quality'])]
        
        if alt_frame_cols and alt_score_cols:
            posture = posture.rename(columns={
                alt_frame_cols[0]: 'frame',
                alt_score_cols[0]: 'alignment_score'
            })
        else:
            print("Required posture alignment columns not found")
            return {}
    
    # Calculate posture alignment statistics
    alignment_stats = {
        'mean_score': float(posture['alignment_score'].mean()),
        'min_score': float(posture['alignment_score'].min()),
        'max_score': float(posture['alignment_score'].max()),
        'score_std': float(posture['alignment_score'].std()),
        'frames_analyzed': len(posture),
    }
    
    # Find frames with best and worst posture
    best_frame = int(posture.loc[posture['alignment_score'].idxmax()]['frame'])
    worst_frame = int(posture.loc[posture['alignment_score'].idxmin()]['frame'])
    
    alignment_stats['best_frame'] = best_frame
    alignment_stats['worst_frame'] = worst_frame
    
    # Look for deviation columns
    deviation_cols = [col for col in posture.columns if 'deviation' in col.lower()]
    
    if deviation_cols:
        deviations = {}
        for col in deviation_cols:
            body_part = col.replace('_deviation', '').title()
            deviations[body_part] = float(posture[col].mean())
            
        alignment_stats['deviations'] = deviations
            
    # Create visualization if output path provided
    if output_path:
        plt.figure(figsize=(14, 8))
        
        # Plot alignment score over time
        plt.plot(posture['frame'], posture['alignment_score'], '-', linewidth=2)
        
        # Highlight best and worst frames
        best_score = posture.loc[posture['frame'] == best_frame, 'alignment_score'].values[0] \
            if best_frame in posture['frame'].values else None
        worst_score = posture.loc[posture['frame'] == worst_frame, 'alignment_score'].values[0] \
            if worst_frame in posture['frame'].values else None
            
        if best_score is not None:
            plt.scatter([best_frame], [best_score], color='green', s=100, 
                        label=f'Best Posture (Frame {best_frame})')
                        
        if worst_score is not None:
            plt.scatter([worst_frame], [worst_score], color='red', s=100, 
                        label=f'Worst Posture (Frame {worst_frame})')
        
        plt.title('Posture Alignment Score Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Alignment Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # If we have deviation data, create a radar chart
        if 'deviations' in alignment_stats and deviation_cols:
            # Create radar chart of deviations
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Get deviation data
            body_parts = list(alignment_stats['deviations'].keys())
            values = list(alignment_stats['deviations'].values())
            
            # Compute angle for each body part
            angles = np.linspace(0, 2*np.pi, len(body_parts), endpoint=False).tolist()
            
            # Close the polygon
            values.append(values[0])
            angles.append(angles[0])
            body_parts.append(body_parts[0])
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            # Set labels
            ax.set_thetagrids(np.degrees(angles[:-1]), body_parts[:-1])
            
            plt.title('Posture Deviations by Body Part')
            plt.tight_layout()
            
            # Save radar chart
            radar_path = os.path.splitext(output_path)[0] + '_deviations.png'
            plt.savefig(radar_path, dpi=300)
            plt.close()
    
    return alignment_stats


def create_posture_heatmap(data, output_path=None):
    """
    Create a heatmap visualization of posture data.
    
    Args:
        data: Posture data (dict or DataFrame)
        output_path: Path to save the visualization
        
    Returns:
        Path to saved visualization
    """
    if isinstance(data, dict):
        # Try to extract joint position data for heatmap
        if 'joint_positions' in data:
            positions = data['joint_positions']
        elif 'keypoints' in data:
            positions = data['keypoints']
        else:
            position_keys = [k for k in data.keys() if any(x in k.lower() 
                                                          for x in ['position', 'keypoint', 'coordinate'])]
            if not position_keys:
                print("No joint position data found for heatmap")
                return None
                
            positions = data[position_keys[0]]
            
        # Convert to DataFrame if not already
        if not isinstance(positions, pd.DataFrame):
            positions = pd.DataFrame(positions)
    else:
        positions = data
        
    # Check if we have x/y position columns
    x_cols = [col for col in positions.columns if '_x' in col.lower()]
    y_cols = [col for col in positions.columns if '_y' in col.lower()]
    
    if not x_cols or not y_cols:
        print("No x/y position columns found for heatmap")
        return None
        
    # Create figure
    plt.figure(figsize=(12, 16))
    
    # Create 2D density plots for each joint
    for i, (x_col, y_col) in enumerate(zip(x_cols[:6], y_cols[:6])):  # Limit to first 6 joints
        joint_name = x_col.replace('_x', '')
        
        # Get x, y coordinates
        x = positions[x_col].dropna().values
        y = positions[y_col].dropna().values
        
        if len(x) < 10 or len(y) < 10:
            continue
            
        # Create subplot
        plt.subplot(3, 2, i+1)
        
        # Create density plot
        sns.kdeplot(x=x, y=y, cmap="biomech", fill=True)
        
        plt.title(f'{joint_name.title()} Position Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
    plt.tight_layout()
    
    # Save visualization
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def create_posture_dashboard(data, output_path=None):
    """
    Create a comprehensive posture analysis dashboard.
    
    Args:
        data: Posture data (dict or DataFrame)
        output_path: Path to save the dashboard
        
    Returns:
        Path to saved dashboard
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Process different data components based on the data structure
    if isinstance(data, dict):
        # Joint angles
        if 'joint_angles' in data:
            angles_df = pd.DataFrame(data['joint_angles']) if not isinstance(data['joint_angles'], pd.DataFrame) else data['joint_angles']
            angle_cols = [col for col in angles_df.columns if 'angle' in col.lower()]
            
            if angle_cols:
                ax1 = fig.add_subplot(gs[0, 0])
                for col in angle_cols[:3]:  # Limit to 3 for clarity
                    ax1.plot(angles_df.index, angles_df[col], label=col.replace('_angle', '').title())
                ax1.set_title('Joint Angles')
                ax1.set_xlabel('Frame')
                ax1.set_ylabel('Angle (degrees)')
                ax1.legend()
                ax1.grid(True)
        
        # Posture alignment
        if 'posture' in data or 'alignment' in data:
            posture_key = 'posture' if 'posture' in data else 'alignment'
            posture_df = pd.DataFrame(data[posture_key]) if not isinstance(data[posture_key], pd.DataFrame) else data[posture_key]
            score_col = next((col for col in posture_df.columns if 'score' in col.lower()), None)
            
            if score_col:
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.plot(posture_df.index, posture_df[score_col])
                ax2.set_title('Posture Alignment Score')
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Score')
                ax2.grid(True)
                
        # Biomechanical efficiency 
        if 'efficiency' in data:
            eff_df = pd.DataFrame(data['efficiency']) if not isinstance(data['efficiency'], pd.DataFrame) else data['efficiency']
            eff_cols = [col for col in eff_df.columns if any(x in col.lower() for x in ['efficiency', 'economy', 'cost'])]
            
            if eff_cols:
                ax3 = fig.add_subplot(gs[1, 0])
                for col in eff_cols[:2]:  # Limit to 2 for clarity
                    ax3.plot(eff_df.index, eff_df[col], label=col.replace('_', ' ').title())
                ax3.set_title('Biomechanical Efficiency')
                ax3.set_xlabel('Frame')
                ax3.set_ylabel('Efficiency')
                ax3.legend()
                ax3.grid(True)
                
        # Deviations
        deviation_data = {}
        for key in data.keys():
            if 'deviation' in key.lower():
                deviation_data[key] = data[key]
                
        if deviation_data or ('posture' in data and any('deviation' in col for col in data['posture'].columns)):
            if deviation_data:
                # If we have separate deviation data
                deviation_df = pd.DataFrame(deviation_data)
            else:
                # Extract from posture data
                deviation_df = pd.DataFrame()
                for col in data['posture'].columns:
                    if 'deviation' in col.lower():
                        deviation_df[col] = data['posture'][col]
            
            if not deviation_df.empty:
                ax4 = fig.add_subplot(gs[1, 1])
                for col in deviation_df.columns:
                    ax4.plot(deviation_df.index, deviation_df[col], 
                             label=col.replace('_deviation', '').title())
                ax4.set_title('Posture Deviations')
                ax4.set_xlabel('Frame')
                ax4.set_ylabel('Deviation')
                ax4.legend()
                ax4.grid(True)
                
        # Joint positions
        position_data = None
        for key in ['joint_positions', 'keypoints', 'coordinates']:
            if key in data:
                position_data = data[key]
                break
                
        if position_data is not None:
            pos_df = pd.DataFrame(position_data) if not isinstance(position_data, pd.DataFrame) else position_data
            
            # Get columns for x and y coordinates
            x_cols = [col for col in pos_df.columns if '_x' in col.lower()]
            y_cols = [col for col in pos_df.columns if '_y' in col.lower()]
            
            if x_cols and y_cols:
                # Get variance of key joint positions
                variance_data = {}
                for x_col, y_col in zip(x_cols, y_cols):
                    joint = x_col.replace('_x', '')
                    x_var = pos_df[x_col].var()
                    y_var = pos_df[y_col].var()
                    total_var = x_var + y_var
                    variance_data[joint] = total_var
                
                # Plot joint position variance
                ax5 = fig.add_subplot(gs[2, 0])
                joints = list(variance_data.keys())
                variances = list(variance_data.values())
                
                # Sort by variance
                sorted_idx = np.argsort(variances)[::-1]  # Descending
                sorted_joints = [joints[i] for i in sorted_idx]
                sorted_vars = [variances[i] for i in sorted_idx]
                
                # Plot top 8 joints by variance
                top_n = min(8, len(sorted_joints))
                bars = ax5.bar(sorted_joints[:top_n], sorted_vars[:top_n])
                
                # Color bars
                norm = plt.Normalize(min(sorted_vars[:top_n]), max(sorted_vars[:top_n]))
                colors = plt.cm.viridis(norm(sorted_vars[:top_n]))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax5.set_title('Joint Position Variance')
                ax5.set_xlabel('Joint')
                ax5.set_ylabel('Variance')
                ax5.set_xticklabels([j.title() for j in sorted_joints[:top_n]], rotation=45, ha='right')
                
        # Summary metrics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # Calculate summary metrics
        metrics = {}
        
        # Average posture score
        if 'posture' in data and score_col:
            metrics['Average Posture Score'] = f"{posture_df[score_col].mean():.1f}/10"
            
        # Peak joint angles
        if 'joint_angles' in data and angle_cols:
            for col in angle_cols[:3]:  # Top 3 joints
                joint = col.replace('_angle', '').title()
                metrics[f"{joint} ROM"] = f"{angles_df[col].max() - angles_df[col].min():.1f}°"
                
        # Overall assessment
        if 'posture' in data and score_col:
            avg_score = posture_df[score_col].mean()
            if avg_score >= 7.5:
                assessment = "Excellent"
                color = 'green'
            elif avg_score >= 6.0:
                assessment = "Good"
                color = 'lightgreen'
            elif avg_score >= 4.0:
                assessment = "Fair"
                color = 'orange'
            else:
                assessment = "Poor"
                color = 'red'
                
            metrics['Overall Assessment'] = assessment
            
        # Summarize joint position stability
        if position_data is not None and x_cols and y_cols:
            stability_scores = []
            # Calculate stability for key joints (lower is more stable)
            for joint_pair in zip(x_cols, y_cols):
                x_col, y_col = joint_pair
                if 'spine' in x_col.lower() or 'torso' in x_col.lower() or 'hip' in x_col.lower():
                    joint = x_col.replace('_x', '')
                    stability = np.sqrt(pos_df[x_col].var() + pos_df[y_col].var())
                    stability_scores.append((joint, stability))
            
            if stability_scores:
                # Lower scores are more stable
                avg_stability = np.mean([s[1] for s in stability_scores])
                if avg_stability < 10:
                    metrics['Posture Stability'] = "High"
                elif avg_stability < 30:
                    metrics['Posture Stability'] = "Moderate"
                else:
                    metrics['Posture Stability'] = "Low"
        
        # Display metrics as a formatted table
        if metrics:
            # Create text summary
            metrics_text = "SUMMARY METRICS\n" + "-" * 40 + "\n"
            for k, v in metrics.items():
                metrics_text += f"{k}: {v}\n"
                
            # Add assessment with colored box if we have it
            if 'Overall Assessment' in metrics:
                y_pos = 0.5  # Center of axes
                assessment = metrics['Overall Assessment']
                
                if assessment == "Excellent":
                    color = "#2ECC71"  # Green
                elif assessment == "Good":
                    color = "#F39C12"  # Orange
                elif assessment == "Fair":
                    color = "#E67E22"  # Light orange
                else:
                    color = "#E74C3C"  # Red
                    
                rect = patches.Rectangle((0.1, y_pos-0.1), 0.8, 0.2, 
                                         facecolor=color, alpha=0.3)
                ax6.add_patch(rect)
                
            # Add metrics text
            ax6.text(0.5, 0.95, "POSTURE ANALYSIS SUMMARY", 
                     ha='center', va='top', fontsize=14, fontweight='bold')
            ax6.text(0.5, 0.85, "-" * 30, ha='center', va='top')
            
            # Add each metric in a formatted way
            y_start = 0.75
            y_step = 0.08
            for i, (key, value) in enumerate(metrics.items()):
                y_pos = y_start - i * y_step
                ax6.text(0.1, y_pos, key + ":", ha='left', va='center', fontweight='bold')
                ax6.text(0.7, y_pos, value, ha='left', va='center')
    
    plt.tight_layout()
    
    # Save dashboard
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def process_posture_file(file_path, output_dir=None):
    """
    Process a single posture data file and generate visualizations.
    
    Args:
        file_path: Path to the posture data file
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of analysis results
    """
    print(f"Processing posture file: {file_path}")
    
    # Determine output directory
    if not output_dir:
        base_dir = os.path.dirname(file_path)
        filename = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(base_dir, f"{filename}_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    try:
        data = load_posture_data(file_path)
    except Exception as e:
        print(f"Error loading posture data: {str(e)}")
        return {}
    
    # Run analyses
    results = {
        "file": file_path,
        "output_dir": output_dir
    }
    
    # Joint angle analysis
    joint_angle_path = os.path.join(output_dir, "joint_angles.png")
    angle_stats = analyze_joint_angles(data, joint_angle_path)
    results["joint_angles"] = angle_stats
    
    # Posture alignment analysis
    alignment_path = os.path.join(output_dir, "posture_alignment.png")
    alignment_stats = analyze_posture_alignment(data, alignment_path)
    results["posture_alignment"] = alignment_stats
    
    # Posture heatmap
    heatmap_path = os.path.join(output_dir, "posture_heatmap.png")
    heatmap_result = create_posture_heatmap(data, heatmap_path)
    results["heatmap_path"] = heatmap_path if heatmap_result else None
    
    # Dashboard
    dashboard_path = os.path.join(output_dir, "posture_dashboard.png")
    dashboard_result = create_posture_dashboard(data, dashboard_path)
    results["dashboard_path"] = dashboard_path if dashboard_result else None
    
    # Save results as JSON
    results_path = os.path.join(output_dir, "posture_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed posture analysis. Results saved to {output_dir}")
    return results


def process_posture_directory(directory, output_dir=None, pattern="*"):
    """
    Process all posture data files in a directory.
    
    Args:
        directory: Directory containing posture data files
        output_dir: Directory to save outputs
        pattern: File pattern to match
        
    Returns:
        List of analysis result dictionaries
    """
    # Find data files
    json_files = find_posture_data_files(directory, f"{pattern}.json")
    csv_files = find_posture_data_files(directory, f"{pattern}.csv")
    npy_files = find_posture_data_files(directory, f"{pattern}.npy")
    
    all_files = json_files + csv_files + npy_files
    
    if not all_files:
        print(f"No posture data files found in {directory} matching pattern {pattern}")
        return []
    
    print(f"Found {len(all_files)} posture data files to process")
    
    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(directory, "posture_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    results = []
    for file_path in all_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        file_output_dir = os.path.join(output_dir, filename)
        result = process_posture_file(file_path, file_output_dir)
        results.append(result)
    
    # Create a summary of all files
    summary = {
        "processed_files": len(results),
        "file_paths": all_files,
        "output_dir": output_dir
    }
    
    summary_path = os.path.join(output_dir, "posture_analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processed {len(results)} posture data files. Summary saved to {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze biomechanical posture data.')
    
    parser.add_argument('--data-dir', '-d', type=str, default='public/results/posture',
                        help='Directory containing posture data files (default: public/results/posture)')
    
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Directory to save analysis results')
    
    parser.add_argument('--file', '-f', type=str,
                        help='Process a specific posture data file')
    
    parser.add_argument('--pattern', '-p', type=str, default='*',
                        help='File pattern to match (default: *)')
    
    args = parser.parse_args()
    
    # Set up visualization style
    setup_visualization_style()
    
    if args.file:
        # Process single file
        process_posture_file(args.file, args.output_dir)
    else:
        # Process directory
        process_posture_directory(args.data_dir, args.output_dir, args.pattern)


if __name__ == "__main__":
    main() 