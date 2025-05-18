#!/usr/bin/env python3
"""
Generate Visualizations - Moriarty

This is a master script that uses all the visualization tools to generate
comprehensive biomechanical analysis visualizations from data in the
public/results folder.
"""

import os
import argparse
import subprocess
import json
import sys
from pathlib import Path
import time
import datetime

# Add scripts directory to path if needed
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Try to import the visualization modules
try:
    import analyze_gifs
    import analyze_posture
    import visualize_results
    MODULES_IMPORTED = True
except ImportError:
    MODULES_IMPORTED = False


def setup_directories(base_dir='public/results', output_dir=None):
    """
    Setup the directory structure for visualization outputs.
    
    Args:
        base_dir: Base directory containing result data
        output_dir: Directory to save visualization outputs
        
    Returns:
        Dictionary with output directory paths
    """
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"visualizations_{timestamp}")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different visualization types
    dirs = {
        'main': output_dir,
        'biomechanics': os.path.join(output_dir, 'biomechanics'),
        'posture': os.path.join(output_dir, 'posture'),
        'gifs': os.path.join(output_dir, 'gif_analysis'),
        'combined': os.path.join(output_dir, 'combined'),
    }
    
    # Create all directories
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    return dirs


def run_visualization_script(script_path, args):
    """
    Run a visualization script using subprocess.
    
    Args:
        script_path: Path to the script
        args: Arguments to pass to the script
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [sys.executable, script_path] + args
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        # Check if successful
        if process.returncode != 0:
            print(f"Error running {script_path}")
            print(process.stderr.read())
            return False
            
        return True
    except Exception as e:
        print(f"Error running {script_path}: {str(e)}")
        return False


def process_biomechanics_data(base_dir, output_dir):
    """
    Process biomechanical data and generate visualizations.
    
    Args:
        base_dir: Base directory containing result data
        output_dir: Directory to save outputs
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*50)
    print("PROCESSING BIOMECHANICS DATA")
    print("="*50)
    
    if MODULES_IMPORTED:
        print("Using imported module...")
        try:
            # Use the imported module directly
            visualize_results.process_results_folder(base_dir, output_dir)
            return True
        except Exception as e:
            print(f"Error using imported module: {str(e)}")
            # Fall back to subprocess method
    
    # Determine the script path
    script_path = os.path.join(SCRIPTS_DIR, "visualize_results.py")
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    args = [
        "--results-dir", base_dir,
        "--output-dir", output_dir
    ]
    
    return run_visualization_script(script_path, args)


def process_posture_data(base_dir, output_dir):
    """
    Process posture data and generate visualizations.
    
    Args:
        base_dir: Base directory containing posture data
        output_dir: Directory to save outputs
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*50)
    print("PROCESSING POSTURE DATA")
    print("="*50)
    
    posture_dir = os.path.join(base_dir, "posture")
    if not os.path.exists(posture_dir):
        print(f"Posture directory not found at {posture_dir}")
        return False
    
    if MODULES_IMPORTED:
        print("Using imported module...")
        try:
            # Use the imported module directly
            analyze_posture.setup_visualization_style()
            analyze_posture.process_posture_directory(posture_dir, output_dir)
            return True
        except Exception as e:
            print(f"Error using imported module: {str(e)}")
            # Fall back to subprocess method
    
    # Determine the script path
    script_path = os.path.join(SCRIPTS_DIR, "analyze_posture.py")
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    args = [
        "--data-dir", posture_dir,
        "--output-dir", output_dir
    ]
    
    return run_visualization_script(script_path, args)


def process_gif_data(base_dir, output_dir):
    """
    Process GIF data and generate visualizations.
    
    Args:
        base_dir: Base directory containing GIF data
        output_dir: Directory to save outputs
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*50)
    print("PROCESSING GIF DATA")
    print("="*50)
    
    gif_dir = os.path.join(base_dir, "gif")
    if not os.path.exists(gif_dir):
        print(f"GIF directory not found at {gif_dir}")
        return False
    
    if MODULES_IMPORTED:
        print("Using imported module...")
        try:
            # Use the imported module directly
            analyze_gifs.process_multiple_gifs(gif_dir, "*.gif", output_dir)
            return True
        except Exception as e:
            print(f"Error using imported module: {str(e)}")
            # Fall back to subprocess method
    
    # Determine the script path
    script_path = os.path.join(SCRIPTS_DIR, "analyze_gifs.py")
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Run the script
    args = [
        "--gif-dir", gif_dir,
        "--output-dir", output_dir
    ]
    
    return run_visualization_script(script_path, args)


def create_combined_dashboard(output_dirs, combined_dir):
    """
    Create a combined dashboard with key visualizations from all analyses.
    
    Args:
        output_dirs: Dictionary with output directory paths
        combined_dir: Directory to save the combined dashboard
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*50)
    print("CREATING COMBINED DASHBOARD")
    print("="*50)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import glob
        from PIL import Image
        
        # Create a figure for the combined dashboard
        plt.figure(figsize=(24, 18))
        grid = GridSpec(3, 3, figure=plt.gcf())
        
        # Find the best visualizations to include
        visualizations = []
        
        # 1. Biomechanics - joint angles
        joint_angle_plots = glob.glob(os.path.join(output_dirs['biomechanics'], "*", "*joint_angles.png"))
        if joint_angle_plots:
            visualizations.append(("Joint Angles", joint_angle_plots[0], grid[0, 0]))
        
        # 2. Biomechanics - velocity profiles
        velocity_plots = glob.glob(os.path.join(output_dirs['biomechanics'], "*", "*velocity*.png"))
        if velocity_plots:
            visualizations.append(("Velocity Profile", velocity_plots[0], grid[0, 1]))
        
        # 3. Posture - alignment
        posture_plots = glob.glob(os.path.join(output_dirs['posture'], "*", "*alignment.png"))
        if posture_plots:
            visualizations.append(("Posture Alignment", posture_plots[0], grid[0, 2]))
        
        # 4. Biomechanics - stride parameters
        stride_plots = glob.glob(os.path.join(output_dirs['biomechanics'], "*", "*stride*.png"))
        if stride_plots:
            visualizations.append(("Stride Parameters", stride_plots[0], grid[1, 0]))
        
        # 5. Posture - joint angle ranges
        angle_range_plots = glob.glob(os.path.join(output_dirs['posture'], "*", "joint_angles.png"))
        if angle_range_plots:
            visualizations.append(("Joint ROM", angle_range_plots[0], grid[1, 1]))
        
        # 6. GIF analysis - position plot
        position_plots = glob.glob(os.path.join(output_dirs['gifs'], "*", "*position.png"))
        if position_plots:
            visualizations.append(("Position Analysis", position_plots[0], grid[1, 2]))
        
        # 7. Posture - heatmap
        heatmap_plots = glob.glob(os.path.join(output_dirs['posture'], "*", "*heatmap.png"))
        if heatmap_plots:
            visualizations.append(("Position Heatmap", heatmap_plots[0], grid[2, 0:2]))
        
        # 8. Posture dashboard
        dashboard_plots = glob.glob(os.path.join(output_dirs['posture'], "*", "*dashboard.png"))
        if dashboard_plots:
            visualizations.append(("Posture Dashboard", dashboard_plots[0], grid[2, 2]))
        
        # Add visualizations to the figure
        for title, img_path, subplot_pos in visualizations:
            try:
                img = Image.open(img_path)
                ax = plt.subplot(subplot_pos)
                ax.imshow(img)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.axis('off')
            except Exception as e:
                print(f"Error adding {title}: {str(e)}")
        
        # Add overall title
        plt.suptitle("Moriarty Biomechanical Analysis Dashboard", fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the combined dashboard
        combined_path = os.path.join(combined_dir, "combined_dashboard.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined dashboard saved to {combined_path}")
        return True
    except Exception as e:
        print(f"Error creating combined dashboard: {str(e)}")
        return False


def generate_html_report(output_dirs, combined_dir):
    """
    Generate an HTML report with all visualizations.
    
    Args:
        output_dirs: Dictionary with output directory paths
        combined_dir: Directory to save the HTML report
        
    Returns:
        Path to the HTML report
    """
    print("\n" + "="*50)
    print("GENERATING HTML REPORT")
    print("="*50)
    
    try:
        # Find all generated visualizations
        visualizations = []
        
        # Combined dashboard
        dashboard_path = os.path.join(combined_dir, "combined_dashboard.png")
        if os.path.exists(dashboard_path):
            visualizations.append({
                "title": "Combined Dashboard",
                "path": os.path.relpath(dashboard_path, output_dirs['main']),
                "section": "dashboard"
            })
        
        # Biomechanics visualizations
        for subdir in os.listdir(output_dirs['biomechanics']):
            subdir_path = os.path.join(output_dirs['biomechanics'], subdir)
            if os.path.isdir(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(subdir_path, img_file)
                        vis_type = ""
                        if "angle" in img_file:
                            vis_type = "Joint Angles"
                        elif "velocity" in img_file:
                            vis_type = "Velocity"
                        elif "stride" in img_file:
                            vis_type = "Stride"
                        elif "grf" in img_file:
                            vis_type = "GRF"
                        elif "power" in img_file:
                            vis_type = "Power"
                        elif "dashboard" in img_file:
                            vis_type = "Dashboard"
                        else:
                            vis_type = "Other"
                            
                        visualizations.append({
                            "title": f"{vis_type} - {subdir}",
                            "path": os.path.relpath(img_path, output_dirs['main']),
                            "section": "biomechanics"
                        })
        
        # Posture visualizations
        for subdir in os.listdir(output_dirs['posture']):
            subdir_path = os.path.join(output_dirs['posture'], subdir)
            if os.path.isdir(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(subdir_path, img_file)
                        vis_type = ""
                        if "angle" in img_file:
                            vis_type = "Joint Angles"
                        elif "alignment" in img_file:
                            vis_type = "Alignment"
                        elif "heatmap" in img_file:
                            vis_type = "Heatmap"
                        elif "dashboard" in img_file:
                            vis_type = "Dashboard"
                        elif "deviation" in img_file:
                            vis_type = "Deviations"
                        else:
                            vis_type = "Other"
                            
                        visualizations.append({
                            "title": f"{vis_type} - {subdir}",
                            "path": os.path.relpath(img_path, output_dirs['main']),
                            "section": "posture"
                        })
        
        # GIF analysis visualizations
        for subdir in os.listdir(output_dirs['gifs']):
            subdir_path = os.path.join(output_dirs['gifs'], subdir)
            if os.path.isdir(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(subdir_path, img_file)
                        vis_type = ""
                        if "position" in img_file:
                            vis_type = "Position"
                        elif "keypoint" in img_file:
                            vis_type = "Keypoints"
                        elif "height" in img_file:
                            vis_type = "Height"
                        else:
                            vis_type = "Other"
                            
                        visualizations.append({
                            "title": f"{vis_type} - {subdir}",
                            "path": os.path.relpath(img_path, output_dirs['main']),
                            "section": "gifs"
                        })
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Moriarty Biomechanical Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                header {{
                    background-color: #2C3E50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                h1, h2, h3 {{
                    color: #2C3E50;
                }}
                .dashboard {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .dashboard img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .section {{
                    margin: 40px 0;
                    border-top: 2px solid #eee;
                    padding-top: 20px;
                }}
                .visualization {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .nav {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .nav a {{
                    margin-right: 15px;
                    color: #2C3E50;
                    text-decoration: none;
                    font-weight: bold;
                }}
                .nav a:hover {{
                    text-decoration: underline;
                }}
                .timestamp {{
                    font-style: italic;
                    color: #777;
                    text-align: center;
                    margin-top: 40px;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>Moriarty Biomechanical Analysis Report</h1>
                <p>Comprehensive visualization and analysis of biomechanical data</p>
            </header>
            
            <div class="nav">
                <a href="#dashboard">Dashboard</a>
                <a href="#biomechanics">Biomechanics</a>
                <a href="#posture">Posture</a>
                <a href="#gifs">GIF Analysis</a>
            </div>
            
            <section id="dashboard" class="section">
                <h2>Dashboard</h2>
                <div class="dashboard">
        """
        
        # Add dashboard image
        dashboard_vis = [v for v in visualizations if v["section"] == "dashboard"]
        if dashboard_vis:
            html_content += f"""
                    <img src="{dashboard_vis[0]['path']}" alt="Combined Dashboard">
                    <p>Combined visualization dashboard showing key metrics and analyses</p>
            """
        
        # Add biomechanics section
        html_content += f"""
                </div>
            </section>
            
            <section id="biomechanics" class="section">
                <h2>Biomechanics Analysis</h2>
        """
        
        biomech_vis = [v for v in visualizations if v["section"] == "biomechanics"]
        for vis in biomech_vis:
            html_content += f"""
                <div class="visualization">
                    <h3>{vis['title']}</h3>
                    <img src="{vis['path']}" alt="{vis['title']}">
                </div>
            """
        
        # Add posture section
        html_content += f"""
            </section>
            
            <section id="posture" class="section">
                <h2>Posture Analysis</h2>
        """
        
        posture_vis = [v for v in visualizations if v["section"] == "posture"]
        for vis in posture_vis:
            html_content += f"""
                <div class="visualization">
                    <h3>{vis['title']}</h3>
                    <img src="{vis['path']}" alt="{vis['title']}">
                </div>
            """
        
        # Add GIF analysis section
        html_content += f"""
            </section>
            
            <section id="gifs" class="section">
                <h2>GIF Analysis</h2>
        """
        
        gif_vis = [v for v in visualizations if v["section"] == "gifs"]
        for vis in gif_vis:
            html_content += f"""
                <div class="visualization">
                    <h3>{vis['title']}</h3>
                    <img src="{vis['path']}" alt="{vis['title']}">
                </div>
            """
        
        # Complete the HTML
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content += f"""
            </section>
            
            <div class="timestamp">
                Report generated on {timestamp}
            </div>
        </body>
        </html>
        """
        
        # Write the HTML to a file
        html_path = os.path.join(combined_dir, "analysis_report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {html_path}")
        return html_path
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive biomechanical visualizations.')
    
    parser.add_argument('--data-dir', '-d', type=str, default='public/results',
                        help='Directory containing result data (default: public/results)')
    
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Directory to save visualizations (default: auto-generated)')
    
    parser.add_argument('--biomechanics-only', action='store_true',
                        help='Only process biomechanics data')
    
    parser.add_argument('--posture-only', action='store_true',
                        help='Only process posture data')
    
    parser.add_argument('--gifs-only', action='store_true',
                        help='Only process GIF data')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Setup directories
    output_dirs = setup_directories(args.data_dir, args.output_dir)
    
    print(f"Output directories created:")
    for name, path in output_dirs.items():
        print(f"  {name}: {path}")
    
    # Process data based on arguments
    process_all = not (args.biomechanics_only or args.posture_only or args.gifs_only)
    
    success = []
    
    if process_all or args.biomechanics_only:
        status = process_biomechanics_data(args.data_dir, output_dirs['biomechanics'])
        success.append(status)
    
    if process_all or args.posture_only:
        status = process_posture_data(args.data_dir, output_dirs['posture'])
        success.append(status)
    
    if process_all or args.gifs_only:
        status = process_gif_data(args.data_dir, output_dirs['gifs'])
        success.append(status)
    
    # Create combined dashboard if multiple analyses were run
    if len(success) > 1 and any(success):
        create_combined_dashboard(output_dirs, output_dirs['combined'])
        
    # Generate HTML report
    html_path = generate_html_report(output_dirs, output_dirs['combined'])
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print("\n" + "="*50)
    print(f"VISUALIZATION GENERATION COMPLETE")
    print(f"Time elapsed: {int(minutes)} minutes, {seconds:.2f} seconds")
    print("="*50)
    
    if html_path:
        print(f"\nView the complete analysis report at: {html_path}")
    
    print(f"\nAll visualizations saved to: {output_dirs['main']}")


if __name__ == "__main__":
    main() 