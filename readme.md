# Face Detection and Emotion Recognition

A real-time face detection application using OpenCV and FER (Facial Expression Recognition) that can detect faces, eyes, and emotions from your webcam feed.

## Features

- Real-time face and eye detection
- Emotion recognition with confidence scores
- FPS (Frames Per Second) counter
- Screenshot and video recording capabilities
- Interactive controls panel with adjustable parameters
- Brightness and contrast adjustment
- Customizable detection visualization
- Live face count display
- Emotion history tracking

## Requirements

- Python >= 3.6
- Webcam
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Set up virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python main.py
```

### Controls:
- 'Q': Quit application
- 'S': Save screenshot
- 'T': Toggle detection visualization
- 'M': Toggle menu overlay
- 'R': Toggle video recording
- 'C': Toggle control panel window
- '+/-': Adjust brightness
- '{/}': Adjust contrast

### Control Panel
Press 'C' to open the control panel which includes:
- Brightness slider (-100 to +100)
- Contrast slider (0.1 to 3.0)
- Detection confidence threshold slider

## Emotion Detection

The application can detect the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Each detected face will display the dominant emotion and its confidence score.

## Performance Notes

- FPS may vary depending on your hardware
- Emotion detection is more computationally intensive than basic face detection
- For better performance, ensure good lighting conditions

## Output Files
- Screenshots: `screenshot_YYYYMMDD_HHMMSS.jpg`
- Video recordings: `recording_YYYYMMDD_HHMMSS.avi`

## Development

To update requirements.txt after installing new packages:
```bash
pip freeze > requirements.txt
```

## Contributing

Feel free to submit issues and enhancement requests.
