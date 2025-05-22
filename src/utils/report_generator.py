import os
import datetime
import jinja2
from src.utils.file_helpers import ensure_directory
from src.utils.visualization import generate_analysis_plots

def generate_sprint_report(analysis_results, output_path, template_path=None):
    """Generate HTML report with analysis results and visualizations"""
    # Generate plots for the report
    plots_dir = os.path.join(os.path.dirname(output_path), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_paths = generate_analysis_plots(analysis_results, plots_dir)
    
    # Get default template if not provided
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'sprint_report_template.html')
    
    # Load template
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)
    
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    
    # Prepare context for template
    context = {
        'title': 'Sprint Analysis Report',
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_results': analysis_results,
        'plot_paths': [os.path.relpath(p, os.path.dirname(output_path)) for p in plot_paths],
    }
    
    # Render template
    html_content = template.render(**context)
    
    # Save HTML report
    ensure_directory(output_path)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path
