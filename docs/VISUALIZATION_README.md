# Moriarty Framework - Results Visualization Suite

This directory contains comprehensive visualization tools designed to showcase the advanced capabilities of the Moriarty sports video analysis framework.

## 🎯 Overview

The Moriarty Framework is a cutting-edge sports video analysis system that demonstrates:

- **Real-time pose detection** with MediaPipe integration
- **Multi-person tracking** and simultaneous analysis
- **3D spatial pose estimation** with depth information
- **Advanced AI confidence scoring** for quality assessment
- **Movement pattern analysis** and biomechanical insights
- **Temporal motion smoothing** and trajectory optimization
- **Sport-specific performance metrics** calculation

## 📊 Visualization Scripts

### 1. `showcase_moriarty.py` (Recommended)
**Quick and impressive visualization showcasing key framework capabilities**

```bash
python showcase_moriarty.py
```

**Features:**
- 📈 Detection performance analysis
- 🧠 AI confidence distribution
- 👥 Multi-person tracking capabilities
- 🏃‍♂️ Movement analysis visualization
- ⚖️ Pose stability assessment
- 📋 Comprehensive performance summary

**Output:** 
- High-resolution showcase image (`moriarty_showcase_YYYYMMDD_HHMMSS.png`)
- Performance report (`moriarty_showcase_report_YYYYMMDD_HHMMSS.txt`)

### 2. `visualize_results.py` (Advanced)
**Comprehensive analysis suite with advanced 3D visualizations**

```bash
python visualize_results.py
```

**Features:**
- 🎨 Interactive capabilities dashboard
- 🎭 3D pose analysis visualization
- 📊 Detailed biomechanical metrics
- 📋 Comprehensive analysis reports

**Output:**
- Capabilities dashboard (PNG)
- Interactive 3D analysis (HTML)
- Detailed technical report (TXT)

### 3. `run_showcase.py` (Helper)
**Quick runner with dependency checking**

```bash
python run_showcase.py
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install numpy matplotlib seaborn pandas scipy plotly
```

Or use the requirements file:
```bash
pip install -r requirements_viz.txt
```

### Data Requirements
Ensure your `models/` directory contains JSON pose data files with the naming pattern:
```
models/
├── athlete_name_pose_data.json
├── another_athlete_pose_data.json
└── ...
```

## 📋 Data Format

The visualization scripts expect JSON files with the following structure:

```json
{
  "video_info": {
    "filename": "athlete.mp4",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "total_frames": 500
  },
  "pose_data": [
    {
      "frame": 1,
      "timestamp": 0.033,
      "poses": [
        {
          "landmarks": [
            {
              "x": 0.5,
              "y": 0.3,
              "z": -0.1,
              "visibility": 0.95
            }
            // ... 33 landmarks total
          ],
          "confidence": 0.87
        }
      ]
    }
    // ... more frames
  ]
}
```

## 🎨 Visualization Features

### Framework Capabilities Showcase
- **Detection Performance**: Real-time pose detection rates
- **AI Confidence**: Advanced confidence scoring visualization  
- **Multi-person Analysis**: Simultaneous tracking capabilities
- **Movement Analysis**: Velocity and acceleration patterns
- **Stability Assessment**: Pose consistency and stability metrics
- **3D Spatial Analysis**: Depth estimation and 3D pose visualization

### Performance Metrics
- Total frames processed
- Detection accuracy rates
- AI confidence distributions
- Multi-person detection capabilities
- Movement velocity analysis
- Pose stability scoring
- Temporal consistency assessment

## 📊 Output Examples

### Showcase Visualization
The main showcase creates a 6-panel visualization demonstrating:

1. **🎯 Real-time Pose Detection** - Detection rate performance
2. **🧠 AI Confidence Distribution** - Confidence scoring analysis
3. **👥 Multi-person Tracking** - Multi-athlete detection capabilities
4. **🏃‍♂️ Movement Analysis** - Velocity and motion patterns
5. **⚖️ Pose Stability** - Stability and consistency metrics
6. **📊 Performance Summary** - Overall framework statistics

### 3D Analysis (Advanced)
Interactive 3D visualization showing:
- Temporal pose evolution
- 3D landmark trajectories
- Spatial movement patterns
- Depth estimation accuracy

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install numpy matplotlib seaborn
   ```

2. **Ensure data is present:**
   ```bash
   ls models/*.json
   ```

3. **Run showcase:**
   ```bash
   python showcase_moriarty.py
   ```

4. **View results:**
   - Check generated PNG visualization
   - Read performance report TXT file

## 🎯 Framework Capabilities Highlighted

### Real-time Processing
- ✅ Live pose detection and tracking
- ✅ Frame-by-frame analysis with temporal consistency
- ✅ Efficient memory management and processing

### Advanced AI Integration
- ✅ MediaPipe pose estimation integration
- ✅ Confidence scoring and quality assessment
- ✅ Multi-person simultaneous detection

### Biomechanical Analysis
- ✅ 3D spatial pose estimation
- ✅ Movement velocity and acceleration calculation
- ✅ Joint angle analysis and stability assessment
- ✅ Sport-specific performance metrics

### Data Quality & Reliability
- ✅ Visibility and occlusion handling
- ✅ Confidence-based filtering
- ✅ Temporal smoothing and consistency checks

## 📈 Performance Benchmarks

The visualization scripts demonstrate the framework's performance across multiple metrics:

- **Detection Rate**: Percentage of frames with successful pose detection
- **AI Confidence**: Average confidence scores from the pose estimation model
- **Multi-person Capability**: Ability to track multiple athletes simultaneously
- **Processing Speed**: Frames per second analysis capability
- **Stability Score**: Pose consistency and tracking reliability

## 🔧 Customization

### Modifying Visualizations
Edit the visualization scripts to:
- Add custom performance metrics
- Modify color schemes and styling
- Include additional analysis features
- Customize output formats

### Adding New Metrics
Extend the analysis by:
- Implementing custom biomechanical calculations
- Adding sport-specific performance indicators
- Including additional statistical analyses
- Creating custom visualization types

## 🆘 Troubleshooting

### Common Issues

1. **No data found**: Ensure JSON files are in the `models/` directory
2. **Import errors**: Install required dependencies with pip
3. **Memory issues**: Use the lightweight `showcase_moriarty.py` script
4. **Display issues**: Ensure matplotlib backend is properly configured

### Performance Tips

- Use `showcase_moriarty.py` for quick demonstrations
- Use `visualize_results.py` for detailed analysis
- Limit data size for faster processing during development
- Save plots to files for sharing and documentation

---

## 🏆 Conclusion

These visualization tools comprehensively demonstrate the Moriarty Framework's advanced capabilities in sports video analysis, showcasing its potential for:

- Professional sports analysis
- Athletic performance optimization
- Biomechanical research
- Real-time coaching applications
- Movement pattern recognition
- Injury prevention analysis

The framework represents a significant advancement in AI-powered sports analysis technology, combining cutting-edge computer vision with practical sports science applications. 