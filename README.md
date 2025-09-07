# OpenCV Joint Tracking Project

This project uses OpenCV for tracking and annotating joints in video frames. It includes functionality for:

- Manual joint annotation
- CSV output for joint coordinates
- Video frame processing

## Requirements

- Python 3.x
- OpenCV (cv2)
- MediaPipe (for automated tracking)

## Files

- `myscript.py`: Manual joint annotation script
- `calcMediapipe.py`: MediaPipe-based joint detection
- Output files:
  - `joints_coordinates.csv`: Manual annotation results
  - `mediapipe_joints.csv`: MediaPipe detection results

## Usage

1. For manual annotation:
   ```bash
   python myscript.py
   ```

2. For MediaPipe-based detection:
   ```bash
   python calcMediapipe.py
   ```
