# Player Tracking and Re-identification System

## Overview

This project implements a comprehensive player tracking and re-identification system for sports video analysis. The system processes a 15-second football video clip and maintains consistent player identities even when players temporarily leave the frame and reappear later.

## Problem Statement

Given a 15-second video clip (`15sec_input_720p.mp4`), the objective is to:
- Identify and track each player throughout the video
- Assign unique player IDs based on initial appearances
- Maintain the same ID when players re-enter the frame after going out of bounds
- Simulate real-time player tracking and re-identification

## My Approach

### 1. Object Detection
- Utilized the provided YOLOv11 model (`best.pt`) fine-tuned for player detection
- Implemented quality-based filtering to reduce false positives
- Added aspect ratio validation for better player detection

### 2. Feature Extraction & Re-identification
I implemented two approaches for player re-identification:

**Traditional Features:**
- Color histograms in HSV space
- Texture features using Sobel gradients
- Spatial layout features

**Deep Learning Approach:**
- Custom Siamese Network with CNN architecture
- 256-dimensional feature extraction
- Cosine similarity for player matching

### 3. Tracking Algorithm
- Hungarian algorithm for optimal assignment
- Multi-criteria matching (position + appearance + IoU)
- Trajectory smoothing with motion prediction
- Limited to 6 players maximum to reduce visual clutter

## System Architecture

```
Input Video → Object Detection → Feature Extraction → Player Matching → Trajectory Tracking → Output
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- GPU recommended but not required (CPU works fine)
- At least 8GB RAM

### Installation

1. **Clone/Download the project:**
```bash
cd Player_Tracking_Assignment
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO loaded successfully')"
```

## File Structure

```
Player_Tracking_Assignment/
├── player_tracker.py          # Main tracking system
├── main.py                   # Primary entry point
├── analysis.py               # Trajectory analysis and visualization
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Technical_Report.md       # Detailed technical documentation
└── output/                   # Results directory
    ├── tracked_output.mp4    # Annotated video
    ├── tracking_results.json # Performance metrics
    └── detailed_tracking.pkl # Raw tracking data
```

## Usage

### Method 1: Basic Tracking (Recommended)
```bash
python main.py --save_video
```


### Method 2: Custom Configuration
```bash
python main.py --input_video ../input_video/15sec_input_720p.mp4 \
               --model_path ../object\ detection\ model/best.pt \
               --output_dir custom_output \
               --confidence 0.6 \
               --save_video
```

## Expected Output

After running the system, you'll get:

1. **Annotated Video:** `output/tracked_output.mp4`
   - Player bounding boxes with IDs
   - Trajectory lines with fading effect
   - Frame information overlay

2. **Trajectory Analysis:** `trajectory_analysis.png`
   - Clean visualization of player movements
   - Statistical analysis of trajectory lengths
   - Professional-quality plots

3. **Performance Analysis:** `performance_analysis.png`
   - Processing time distribution
   - FPS over time
   - Detection statistics

4. **Results Data:** `output/tracking_results.json`
   - Video metadata
   - Performance metrics
   - Player position data

## Configuration Options

### PlayerTracker Parameters
You can modify these in `player_tracker.py`:

```python
self.max_players = 6           # Maximum players to track
self.confidence_threshold = 0.6 # Detection confidence
self.max_disappeared = 30      # Frames before player removal
self.max_distance = 150        # Matching distance threshold
self.iou_threshold = 0.4       # IoU threshold for matching
```

### Processing Optimization
For better performance:
- Use GPU if available (automatic detection)
- Adjust confidence threshold (higher = faster)
- Reduce max_players for speed
- Lower video resolution if needed

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**2. "CUDA out of memory" (if using GPU)**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python main.py --save_video
```

**3. "No video output generated"**
- Make sure to use `--save_video` flag
- Check that input video path is correct
- Verify model file exists

**4. Slow processing (< 1 FPS)**
- Normal for CPU processing
- Consider using GPU
- Reduce confidence threshold
- Process shorter video clips

### Performance Tips

1. **GPU Acceleration:**
   - Install CUDA-compatible PyTorch
   - Verify GPU usage: `nvidia-smi`

2. **Memory Optimization:**
   - Close other applications
   - Use batch processing for multiple videos

3. **Speed vs. Accuracy:**
   - Higher confidence = faster processing
   - Lower max_players = better performance
   - Shorter videos = quicker results

## Results Analysis

### Trajectory Visualization
The system generates clean trajectory plots showing:
- Player movement patterns
- Start and end positions
- Trajectory smoothing effects
- Statistical summaries

### Performance Metrics
- Processing speed (FPS)
- Detection accuracy
- Player tracking stability
- Memory usage statistics

## Limitations & Future Work

### Current Limitations
1. **Processing Speed:** 0.4 FPS on CPU (acceptable for demo)
2. **Player Limit:** Maximum 6 players (design choice for clarity)
3. **Lighting Conditions:** Works best with good lighting
4. **Occlusion Handling:** Partial occlusions may cause temporary ID loss

### Planned Improvements
1. **Roboflow Trackers Integration:** Exploring the new Trackers library
2. **Kalman Filtering:** Better motion prediction
3. **Multi-scale Detection:** Handle varying player sizes
4. **Real-time Optimization:** GPU acceleration and model quantization

## Dependencies

### Core Libraries
- `ultralytics>=8.0.0` - YOLOv11 model
- `opencv-python>=4.8.0` - Computer vision
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing

### Visualization & Analysis
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `pandas>=2.0.0` - Data analysis

### Utilities
- `tqdm>=4.65.0` - Progress bars
- `scikit-learn>=1.3.0` - Machine learning utilities

## Technical Details

### Siamese Network Architecture
```
Input (128x64x3) → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → 
Conv2D(256) → MaxPool → FC(512) → FC(256) → Feature Vector
```

### Matching Algorithm
1. Extract features for all detected players
2. Compute similarity matrices (position + appearance + IoU)
3. Apply Hungarian algorithm for optimal assignment
4. Update player trajectories and feature histories

### Trajectory Smoothing
- Moving average filter for position smoothing
- Velocity-based motion prediction
- Fade-in trajectory visualization

## Contact & Support

For questions or issues:
1. Check the troubleshooting section
2. Review the Technical_Report.md for detailed explanations
3. Examine the code comments for implementation details

## License

This project is developed for educational purposes as part of a computer vision assignment.

---
