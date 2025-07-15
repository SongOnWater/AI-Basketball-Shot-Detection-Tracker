#!/bin/bash

# Basketball Shot Detection Setup Script
set -e

echo "=== Basketball Shot Detection Setup ==="

# 1. Create conda environment
echo "Creating conda environment..."
conda create -n basketball-ai python=3.8 -y
source activate basketball-ai

# 2. Install Python packages
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install --upgrade ultralytics

# 3. Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y qt5-default libxcb-xinerama0 xvfb

# 4. Verify installation
echo "Verifying installation..."
python -c "import cv2, torch; print('OpenCV version:', cv2.__version__); print('PyTorch version:', torch.__version__)"

# 5. Create run script
echo "Creating run script..."
cat > run_detector.sh << 'EOL'
#!/bin/bash
source activate basketball-ai
cd "$(dirname "$0")"
xvfb-run --auto-servernum --server-args="-screen 0 1280x1024x24" python shot_detector.py
EOL

chmod +x run_detector.sh

echo "=== Setup Complete ==="
echo "To run the basketball shot detector:"
echo "  ./run_detector.sh"
echo ""
echo "For webcam input, make sure your camera is connected."
echo "For video file input, modify shot_detector.py to specify your video path."