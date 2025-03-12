# Basketball Shot Detector

A computer vision application that detects and analyzes basketball shots in real-time using a webcam.

## Features

- Real-time pose detection using MediaPipe
- Basketball shot detection and classification
- Simple visualization of results

## Requirements

- Python 3.9+
- Webcam with decent resolution (720p recommended)
- Consistent lighting conditions

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/basketball_shot_detector.git
cd basketball_shot_detector
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the main application:
```
python -m src.main
```

- The application will start your webcam and display the video feed with pose landmarks.
- When a basketball shot is detected, it will show the shot type on screen.
- Press 'q' to quit the application.

## How It Works

1. **Pose Detection**: MediaPipe's pose estimation model detects body landmarks.
2. **Arm Angle Calculation**: The system calculates the angle between the upper arm and forearm.
3. **Shot Detection**: Changes in arm angle that match shooting patterns are detected.
4. **Shot Classification**: Shots are classified based on arm angle characteristics.

## Limitations

- Works best with a clear view of the person shooting
- Designed for right-handed shooters by default
- Basic shot classification without machine learning

## Future Improvements

- Add support for left-handed shooters
- Implement machine learning for better shot success prediction
- Improve visualization with shot trajectory estimation
- Add recording and statistics tracking features