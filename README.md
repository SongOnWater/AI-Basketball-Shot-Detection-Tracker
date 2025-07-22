# Real-Time AI Basketball Shot Detection with YOLOv8 and OpenCV
Author: [Avi Shah](https://www.linkedin.com/in/-avishah/) (2023)

Score Detection Accuracy: 95% <br>
Shot Detection Accuracy: 97% <br>

https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker/assets/115107522/469750e1-0f3c-4b07-9fc5-5387831742f7

## Introduction

This project combines the power of Machine Learning and Computer Vision for the purpose of detecting and analyzing basketball shots in real-time! Built upon the latest YOLOv8 (You Only Look Once) machine learning model and the OpenCV library, the program can process video streams from various sources, such as live webcam feed or pre-recorded videos, providing a tool that can be used for an immersive playing experience and enhanced game analytics.

## Language
- [English](README.md)
- [中文](README-zh.md)

## Directory Structure

```
├── README.md
├── best.pt             # Pre-trained model
├── config.yaml         # Dataset configuration
├── main.py             # YoloV8 training script
├── requirements.txt    
├── shot_detector.py    # Detection algorithm
├── utils.py            # Helper functions
└── video_test_5.mp4    # Test Video
```

## Model Training

The training process utilizes the ultralytics YOLO implementation and a custom dataset specified in the 'config.yaml' file. The model undergoes a set number of training epochs, with the resulting weights of the best-performing model saved for subsequent usage in shot detection. Although this model worked for my usage, a different dataset or training method might work better for your specific project.

## Algorithm

The core of this project is an algorithm that uses the trained YOLOv8 model to detect basketballs and hoops in each frame. It then analyzes the motion and position of the basketball relative to the hoop to determine if a shot has been made.

To enhance the accuracy of the shot detection, the algorithm not only tracks the ball's position over time but also applies data-cleaning techniques to both the ball and hoop positions. The algorithm is designed to filter out inaccurate data points, remove points beyond a certain frame limit and prevent jumping from one object to another to maintain the accuracy of the detection.

A linear regression is used to predict the ball's trajectory based on its positions. If the projected trajectory intersects with the hoop, the algorithm registers it as a successful shot.

## Installation & Usage

### Quick Start (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker.git
cd AI-Basketball-Shot-Detector-Tracker
```

2. Run the automated setup script:
```bash
chmod +x setup_basketball_ai.sh
./setup_basketball_ai.sh
```

3. Start the shot detector:
```bash
./run_detector.sh
```

### Manual Installation

For advanced users or custom setups:

1. Create conda environment:
```bash
conda create -n basketball-ai python=3.10 
conda activate basketball-ai
```

2. Install Python packages:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y qt5-default libxcb-xinerama0 xvfb
```

4. Run the detector:
```bash
xvfb-run --auto-servernum --server-args="-screen 0 1280x1024x24" python shot_detector.py
```

### Environment Details

This project uses the following key packages:
- Python with the following main dependencies:
  - Ultralytics YOLOv8: 8.3.166
  - PyTorch: 2.7.1
  - OpenCV: 4.11.0.86
  - NumPy: 2.2.6
  - cvzone: 1.5.6

### Using the Pre-trained Model

**If you don't want to train the model yourself**, simply use the provided 'best.pt' model (skip training steps).

### Training Your Own Model (Optional)

1. Download the dataset specified in 'config.yaml'
2. Adjust the paths in the configuration file
3. Follow the instructions in 'main.py' to train the model

### Input Options
- Webcam: Default input (make sure camera is connected)
- Video file: Modify shot_detector.py to specify your video path
   
Contributions to this project are welcome - submit a pull request. For issues or suggestions, open an issue in this repository.

## Disclaimer

The model's performance can vary based on factors such as the quality of the video feed, lighting conditions, and the clarity of the basketball and hoop in the video. Furthermore, this program will **not** work if multiple basketballs and hoops are in frame. For testing, this program had input videos that were shot outdoors from a phone camera on the ground.