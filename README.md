# Object Tracking with ROI Selection and Optical Flow

This Python script uses OpenCV to perform object tracking in a video by allowing the user to select a region of interest (ROI) interactively. It tracks feature points inside the selected ROI using the Lucas-Kanade optical flow method and visualizes their trajectories.

## Features

- Interactive ROI selection on the first frame of the video.
- Detection of good feature points inside the selected ROI using Shi-Tomasi corner detection.
- Tracking of these feature points across frames using Lucas-Kanade optical flow.
- Filtering of tracked points to keep only those inside the original ROI.
- Visualization of tracking paths with lines and points.
- Automatic clearing of tracking lines when no points remain inside the ROI.
- Exit the program by pressing the ESC key.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies via pip if needed:

```bash
pip install opencv-python numpy
```

## Usage

1. Update the video path in the script:

```python
cap = cv2.VideoCapture('/path/to/your/video.mp4')
```

2. Run the script:

```bash
python object_tracking.py
```

3. When the first frame appears, select the object to track by drawing a rectangle around it. Press ENTER or SPACE to confirm.

4. The program will track feature points inside the selected ROI and display the tracking in a window.

5. Press ESC at any time to exit.

## How it works

- The script reads the first frame and lets the user select the ROI.
- It detects good feature points inside the ROI using Shi-Tomasi corner detection.
- Using Lucas-Kanade optical flow, it tracks these points frame by frame.
- Points moving outside the ROI are discarded to avoid drifting.
- When no points remain inside the ROI, the tracking lines are cleared.
- The tracking visualization shows green lines for trajectories and red circles for current points.

## Notes

- Make sure the video file path is correct.
- The tracking quality depends on the selected ROI and video content.
- This script works best for moderate motion and well-defined objects.


