# Moriarty Visualization Scripts

This directory contains scripts for analyzing and visualizing biomechanical data from the Moriarty system.

## Available Scripts

- **generate_visualizations.py**: Master script that runs all visualization tools
- **visualize_results.py**: Creates charts from biomechanical data in JSON/CSV files
- **analyze_posture.py**: Analyzes posture data and creates specialized visualizations
- **analyze_gifs.py**: Extracts and analyzes frames from GIF files

## Quick Start

To generate all visualizations from existing data:

```bash
python scripts/generate_visualizations.py
```

This will:
1. Process all data in `public/results/`
2. Generate comprehensive visualizations
3. Create a combined dashboard
4. Generate an HTML report

## Usage Options

### Master Visualization Script

```bash
python scripts/generate_visualizations.py [OPTIONS]
```

Options:
- `--data-dir/-d`: Directory containing result data (default: public/results)
- `--output-dir/-o`: Directory to save visualizations (default: auto-generated timestamp folder)
- `--biomechanics-only`: Only process biomechanics data
- `--posture-only`: Only process posture data
- `--gifs-only`: Only process GIF data

### Biomechanics Visualization

```bash
python scripts/visualize_results.py [OPTIONS]
```

Options:
- `--results-dir/-r`: Directory containing result files (default: public/results)
- `--output-dir/-o`: Directory to save visualizations
- `--file/-f`: Process a specific result file instead of the whole directory
- `--plot-type/-p`: Type of plot to generate (choices: joint_angles, velocity, stride, grf, power, dashboard, all)

### Posture Analysis

```bash
python scripts/analyze_posture.py [OPTIONS]
```

Options:
- `--data-dir/-d`: Directory containing posture data files (default: public/results/posture)
- `--output-dir/-o`: Directory to save analysis results
- `--file/-f`: Process a specific posture data file
- `--pattern/-p`: File pattern to match (default: *)

### GIF Analysis

```bash
python scripts/analyze_gifs.py [OPTIONS]
```

Options:
- `--gif-dir/-d`: Directory containing GIF files (default: public/results/gif)
- `--output-dir/-o`: Directory to save analysis results
- `--gif-file/-f`: Process a specific GIF file instead of a directory
- `--pattern/-p`: Glob pattern to match GIF files (default: *.gif)

## Example Usage

Process only posture data and save to a specific output directory:

```bash
python scripts/generate_visualizations.py --posture-only --output-dir results/posture_analysis
```

Extract frames and analyze a specific GIF file:

```bash
python scripts/analyze_gifs.py --gif-file public/results/gif/sprint_analysis.gif --output-dir analysis/sprint
```

Generate velocity and stride visualizations for a specific result file:

```bash
python scripts/visualize_results.py --file public/results/athlete1_data.json --plot-type velocity
```

## Dependencies

These scripts require the following libraries:
- numpy
- pandas
- matplotlib
- seaborn
- PIL (Pillow)
- opencv-python (cv2)

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn pillow opencv-python
``` 