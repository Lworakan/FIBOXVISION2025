# GODANGVISION2025

## Overview
The GODANGVISION2025 workspace is designed for finding three-dimensional coordinates using the Intel RealSense D450 depth camera.


## Features
- **3D Coordinate Extraction**: Uses the RealSense D450 to obtain accurate (X, Y, Z) coordinates of objects.
- **Depth Sensing**: Captures depth information to map the environment effectively.
- **High Precision**: Suitable for applications requiring fine-grained depth data.  (Tolerance error is 2 cm)
- **Integration with Machine Vision**: Can be used for object recognition and tracking.


## Installation
```bash
# Install Intel RealSense SDK 2.0
sudo apt update
sudo apt install librealsense2-dev

# Install required Python packages
pip install pyrealsense2 opencv-python numpy

# Clone the repository and navigate to the workspace
git clone [https://github.com/GODANGVISION2025/realsense_workspace.git](https://github.com/Lworakan/GODANGVISION2025.git)
