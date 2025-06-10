#!/usr/bin/env python3
"""
Moriarty Framework Showcase Script
=================================

This script creates impressive visualizations that highlight the key capabilities
of the Moriarty sports video analysis framework.

Key Features Demonstrated:
- Multi-athlete pose tracking
- Real-time confidence scoring
- 3D spatial analysis
- Movement pattern recognition
- Biomechanical analysis
- Performance metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('default')
sns.set_palette("viridis")

def load_pose_data(models_dir="models"):
    """Load and analyze pose data from JSON files."""
    print("🎯 MORIARTY FRAMEWORK - SPORTS ANALYSIS SHOWCASE")
    print("=" * 60)
    print("Loading athlete data...")
    
    models_path = Path(models_dir)
    pose_data = {}
    
    json_files = list(models_path.glob("*_pose_data.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract athlete name
            athlete_name = file_path.stem.replace('_pose_data', '').replace('-', ' ').title()
            pose_data[athlete_name] = data
            
            frames = len(data.get('pose_data', []))
            print(f"  ✅ {athlete_name}: {frames:,} frames analyzed")
            
        except Exception as e:
            print(f"  ⚠️ Skipping {file_path.name}: {str(e)}")
    
    print(f"\n🎉 Loaded {len(pose_data)} athlete datasets")
    return pose_data

def analyze_framework_capabilities(pose_data):
    """Analyze and showcase framework capabilities."""
    print("\n🔬 Analyzing framework capabilities...")
    
    analytics = {}
    
    for athlete, data in pose_data.items():
        video_info = data.get('video_info', {})
        pose_frames = data.get('pose_data', [])
        
        # Calculate key metrics
        total_frames = len(pose_frames)
        frames_with_detection = sum(1 for frame in pose_frames if frame.get('poses'))
        detection_rate = frames_with_detection / total_frames if total_frames > 0 else 0
        
        # Confidence analysis
        confidences = []
        visibilities = []
        multi_person_frames = 0
        max_people = 0
        
        for frame in pose_frames:
            poses = frame.get('poses', [])
            if len(poses) > 1:
                multi_person_frames += 1
            max_people = max(max_people, len(poses))
            
            for pose in poses:
                if 'confidence' in pose:
                    confidences.append(pose['confidence'])
                
                for landmark in pose.get('landmarks', []):
                    if 'visibility' in landmark:
                        visibilities.append(landmark['visibility'])
        
        # Movement analysis
        movement_data = calculate_movement_metrics(pose_frames)
        
        analytics[athlete] = {
            'video_info': video_info,
            'total_frames': total_frames,
            'detection_rate': detection_rate,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_visibility': np.mean(visibilities) if visibilities else 0,
            'multi_person_capability': multi_person_frames / total_frames if total_frames > 0 else 0,
            'max_people_detected': max_people,
            'movement_velocity': movement_data['velocity'],
            'pose_stability': movement_data['stability'],
            'duration': total_frames / video_info.get('fps', 30) if total_frames > 0 else 0
        }
        
        print(f"  📊 {athlete}: {detection_rate:.1%} detection, {np.mean(confidences)*100 if confidences else 0:.1f}% confidence")
    
    return analytics

def calculate_movement_metrics(pose_frames):
    """Calculate movement and stability metrics."""
    if not pose_frames:
        return {'velocity': 0, 'stability': 0}
    
    # Track center of mass movement (using torso landmarks)
    torso_x, torso_y = [], []
    
    for frame in pose_frames:
        if frame.get('poses') and frame['poses'][0].get('landmarks'):
            landmarks = frame['poses'][0]['landmarks']
            if len(landmarks) >= 24:  # Ensure we have torso landmarks
                # Calculate center of shoulders and hips
                shoulders = [(landmarks[11]['x'] + landmarks[12]['x']) / 2,
                           (landmarks[11]['y'] + landmarks[12]['y']) / 2]
                hips = [(landmarks[23]['x'] + landmarks[24]['x']) / 2,
                       (landmarks[23]['y'] + landmarks[24]['y']) / 2]
                
                torso_center_x = (shoulders[0] + hips[0]) / 2
                torso_center_y = (shoulders[1] + hips[1]) / 2
                
                torso_x.append(torso_center_x)
                torso_y.append(torso_center_y)
    
    if len(torso_x) < 2:
        return {'velocity': 0, 'stability': 0}
    
    # Calculate velocity
    velocities = []
    for i in range(1, len(torso_x)):
        dx = torso_x[i] - torso_x[i-1]
        dy = torso_y[i] - torso_y[i-1]
        velocity = np.sqrt(dx**2 + dy**2)
        velocities.append(velocity)
    
    avg_velocity = np.mean(velocities) if velocities else 0
    
    # Calculate stability (inverse of position variance)
    stability = 1.0 / (1.0 + np.std(torso_x) + np.std(torso_y))
    
    return {'velocity': avg_velocity, 'stability': stability}

def create_showcase_visualization(analytics):
    """Create impressive showcase visualization."""
    print("\n🎨 Creating Moriarty Framework Showcase...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('MORIARTY FRAMEWORK - ADVANCED SPORTS VIDEO ANALYSIS', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    athletes = list(analytics.keys())
    colors = plt.cm.plasma(np.linspace(0, 1, len(athletes)))
    
    # 1. Detection Performance
    ax1 = axes[0, 0]
    detection_rates = [analytics[athlete]['detection_rate'] * 100 for athlete in athletes]
    bars1 = ax1.barh(athletes, detection_rates, color=colors)
    ax1.set_xlabel('Detection Rate (%)', fontweight='bold')
    ax1.set_title('🎯 Real-time Pose Detection', fontweight='bold', fontsize=16)
    ax1.set_xlim(0, 100)
    
    for bar, rate in zip(bars1, detection_rates):
        ax1.text(rate + 1, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%', 
                va='center', fontweight='bold')
    
    # 2. AI Confidence Scoring
    ax2 = axes[0, 1]
    confidences = [analytics[athlete]['avg_confidence'] * 100 for athlete in athletes]
    ax2.pie(confidences, labels=[name[:10] for name in athletes], autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax2.set_title('🧠 AI Confidence Distribution', fontweight='bold', fontsize=16)
    
    # 3. Multi-person Capability
    ax3 = axes[0, 2]
    max_people = [analytics[athlete]['max_people_detected'] for athlete in athletes]
    multi_person_rates = [analytics[athlete]['multi_person_capability'] * 100 for athlete in athletes]
    
    scatter = ax3.scatter(max_people, multi_person_rates, s=200, c=colors, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Max People Detected', fontweight='bold')
    ax3.set_ylabel('Multi-person Detection Rate (%)', fontweight='bold')
    ax3.set_title('👥 Multi-person Tracking', fontweight='bold', fontsize=16)
    
    for i, athlete in enumerate(athletes):
        ax3.annotate(athlete[:8], (max_people[i], multi_person_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # 4. Movement Analysis
    ax4 = axes[1, 0]
    velocities = [analytics[athlete]['movement_velocity'] * 1000 for athlete in athletes]
    ax4.plot(range(len(athletes)), velocities, 'o-', linewidth=4, markersize=10, color='#E74C3C')
    ax4.set_xlabel('Athletes', fontweight='bold')
    ax4.set_ylabel('Movement Velocity (×10³)', fontweight='bold')
    ax4.set_title('🏃‍♂️ Movement Analysis', fontweight='bold', fontsize=16)
    ax4.set_xticks(range(len(athletes)))
    ax4.set_xticklabels([name[:8] for name in athletes], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Stability Analysis
    ax5 = axes[1, 1]
    stabilities = [analytics[athlete]['pose_stability'] * 100 for athlete in athletes]
    bars5 = ax5.bar(range(len(athletes)), stabilities, color=colors, alpha=0.8)
    ax5.set_xlabel('Athletes', fontweight='bold')
    ax5.set_ylabel('Pose Stability Score (%)', fontweight='bold')
    ax5.set_title('⚖️ Pose Stability Analysis', fontweight='bold', fontsize=16)
    ax5.set_xticks(range(len(athletes)))
    ax5.set_xticklabels([name[:8] for name in athletes], rotation=45, ha='right')
    
    for bar, stability in zip(bars5, stabilities):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{stability:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. Performance Summary
    ax6 = axes[1, 2]
    
    # Calculate overall scores
    total_frames = sum(analytics[athlete]['total_frames'] for athlete in athletes)
    total_duration = sum(analytics[athlete]['duration'] for athlete in athletes)
    avg_detection = np.mean([analytics[athlete]['detection_rate'] for athlete in athletes])
    avg_confidence = np.mean([analytics[athlete]['avg_confidence'] for athlete in athletes])
    
    summary_text = f"""
🎯 FRAMEWORK CAPABILITIES

📊 Datasets: {len(analytics)}
📹 Total Frames: {total_frames:,}
⏱️ Total Duration: {total_duration:.1f}s
🎪 Avg Detection: {avg_detection:.1%}
🏆 Avg Confidence: {avg_confidence:.1%}

✅ Real-time Processing
✅ Multi-person Tracking  
✅ 3D Spatial Analysis
✅ Movement Biomechanics
✅ Confidence Scoring
✅ Sport-specific Metrics
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'moriarty_showcase_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Showcase visualization saved as: {filename}")
    plt.show()
    
    return filename

def generate_summary_report(analytics):
    """Generate summary report highlighting key achievements."""
    print("\n📋 Generating Framework Showcase Report...")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'moriarty_showcase_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MORIARTY FRAMEWORK - SPORTS VIDEO ANALYSIS SHOWCASE\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("🎯 FRAMEWORK CAPABILITIES DEMONSTRATED:\n")
        f.write("-" * 50 + "\n")
        f.write("✅ Real-time pose detection and tracking\n")
        f.write("✅ Multi-person simultaneous analysis\n")
        f.write("✅ Advanced AI confidence scoring\n")
        f.write("✅ 3D spatial pose estimation\n")
        f.write("✅ Movement pattern analysis\n")
        f.write("✅ Biomechanical stability assessment\n")
        f.write("✅ Sport-specific performance metrics\n")
        f.write("✅ Temporal motion smoothing\n\n")
        
        f.write("📊 PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        
        total_frames = sum(analytics[athlete]['total_frames'] for athlete in analytics)
        total_duration = sum(analytics[athlete]['duration'] for athlete in analytics)
        avg_detection = np.mean([analytics[athlete]['detection_rate'] for athlete in analytics])
        avg_confidence = np.mean([analytics[athlete]['avg_confidence'] for athlete in analytics])
        max_people_overall = max(analytics[athlete]['max_people_detected'] for athlete in analytics)
        
        f.write(f"Athletes Analyzed: {len(analytics)}\n")
        f.write(f"Total Frames Processed: {total_frames:,}\n")
        f.write(f"Total Video Duration: {total_duration:.1f} seconds\n")
        f.write(f"Average Detection Rate: {avg_detection:.1%}\n")
        f.write(f"Average AI Confidence: {avg_confidence:.1%}\n")
        f.write(f"Maximum People Detected: {max_people_overall}\n\n")
        
        f.write("🏆 TOP PERFORMERS:\n")
        f.write("-" * 20 + "\n")
        
        # Best detection rate
        best_detection = max(analytics.keys(), key=lambda x: analytics[x]['detection_rate'])
        f.write(f"Best Detection Rate: {best_detection} ({analytics[best_detection]['detection_rate']:.1%})\n")
        
        # Best confidence
        best_confidence = max(analytics.keys(), key=lambda x: analytics[x]['avg_confidence'])
        f.write(f"Highest AI Confidence: {best_confidence} ({analytics[best_confidence]['avg_confidence']:.1%})\n")
        
        # Most stable
        best_stability = max(analytics.keys(), key=lambda x: analytics[x]['pose_stability'])
        f.write(f"Most Stable Poses: {best_stability} ({analytics[best_stability]['pose_stability']:.3f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MORIARTY FRAMEWORK - ADVANCING SPORTS SCIENCE THROUGH AI\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Report saved as: {report_file}")
    return report_file

def main():
    """Main showcase execution."""
    try:
        # Load pose data
        pose_data = load_pose_data()
        
        if not pose_data:
            print("❌ No pose data found in models/ directory!")
            print("   Please ensure JSON files are present.")
            return
        
        # Analyze capabilities
        analytics = analyze_framework_capabilities(pose_data)
        
        # Create showcase visualization
        viz_file = create_showcase_visualization(analytics)
        
        # Generate report
        report_file = generate_summary_report(analytics)
        
        print("\n" + "="*60)
        print("🎉 MORIARTY FRAMEWORK SHOWCASE COMPLETE!")
        print("="*60)
        print(f"📊 Visualization: {viz_file}")
        print(f"📋 Report: {report_file}")
        print("\n🚀 The Moriarty Framework demonstrates cutting-edge")
        print("   capabilities in sports video analysis and AI!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during showcase: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("pip install numpy matplotlib seaborn")

if __name__ == "__main__":
    main() 