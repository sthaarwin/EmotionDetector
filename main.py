import cv2
import numpy as np
import time
from datetime import datetime
from fer import FER
from collections import deque

# Initialize detectors
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
emotion_detector = FER(mtcnn=False)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print('Unable to load camera.')
    exit()

# Initialize variables
show_detection = True
show_menu = True
recording = False
video_writer = None
brightness = 0
contrast = 1
emotion_history = deque(maxlen=30)  # Store last 30 emotions
prev_frame_time = 0
new_frame_time = 0
show_controls = False
detection_threshold = 0.5

def on_brightness_change(value):
    global brightness
    brightness = value - 100  # Scale from 0-200 to -100-100

def on_contrast_change(value):
    global contrast
    contrast = value / 50  # Scale from 0-150 to 0-3.0

def on_detection_threshold_change(value):
    global detection_threshold
    detection_threshold = value / 100

def create_control_window():
    # Create a window with a fixed size
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 200)
    
    # Create the control panel background
    control_panel = np.ones((200, 400, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Add trackbars
    cv2.createTrackbar('Brightness', 'Controls', 100, 200, on_brightness_change)
    cv2.createTrackbar('Contrast', 'Controls', 50, 150, on_contrast_change)
    cv2.createTrackbar('Detection Confidence', 'Controls', 50, 100, on_detection_threshold_change)
    
    return control_panel

def add_menu_overlay(frame):
    if not show_menu:
        return
    menu_items = [
        "Q: Quit",
        "S: Screenshot",
        "T: Toggle detection",
        "M: Toggle menu",
        "R: Toggle recording",
        "C: Toggle controls",
        "[-/+]: Adjust brightness",
        "[{/}]: Adjust contrast",
    ]
    y = 90
    for item in menu_items:
        cv2.putText(frame, item, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
        y += 25

def adjust_brightness_contrast(frame, brightness=0, contrast=1):
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

# Create control window
create_control_window()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print('Unable to read frame.')
        break
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    
    # Apply brightness and contrast adjustments
    frame = adjust_brightness_contrast(frame, brightness, contrast)
    
    try:
        # Detect emotions with error handling
        emotions = emotion_detector.detect_emotions(frame)
        
        if emotions:
            # Store dominant emotion in history
            dominant_emotion = sorted(emotions[0]['emotions'].items(), key=lambda x: x[1], reverse=True)[0][0]
            emotion_history.append(dominant_emotion)
        
        if show_detection and emotions:
            for emotion in emotions:
                x, y, w, h = emotion['box']
                emotions_dict = emotion['emotions']
                
                # Get top 2 emotions for better accuracy
                sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:2]
                emotion_text = " | ".join([f"{emo}: {score:.2f}" for emo, score in sorted_emotions])
                
                # Draw face rectangle with thicker line
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Improve emotion text visibility
                cv2.putText(frame, emotion_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)  # black outline
                cv2.putText(frame, emotion_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # white text
                
                # Detect eyes only if we have a valid face
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        emotions = []

    # Add debug information
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.putText(frame, f'Faces detected: {len(emotions)}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add menu overlay
    add_menu_overlay(frame)
    
    # Show emotion history
    if emotion_history:
        latest_emotion = max(set(emotion_history), key=emotion_history.count)
        cv2.putText(frame, f'Dominant emotion: {latest_emotion}', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    
    # Handle keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if recording:
            video_writer.release()
        break
    elif key == ord('s'):  # Save screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f'screenshot_{timestamp}.jpg', frame)
    elif key == ord('r'):  # Toggle recording
        if not recording:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_writer = cv2.VideoWriter(f'recording_{timestamp}.avi',
                                         cv2.VideoWriter_fourcc(*'XVID'),
                                         20.0, (frame.shape[1], frame.shape[0]))
            recording = True
        else:
            video_writer.release()
            recording = False
    elif key == ord('m'):  # Toggle menu
        show_menu = not show_menu
    elif key == ord('+'):  # Increase brightness
        brightness = min(brightness + 10, 100)
    elif key == ord('-'):  # Decrease brightness
        brightness = max(brightness - 10, -100)
    elif key == ord('}'):  # Increase contrast
        contrast = min(contrast + 0.1, 3.0)
    elif key == ord('{'):  # Decrease contrast
        contrast = max(contrast - 0.1, 0.1)
    elif key == ord('t'):  # Toggle detection boxes
        show_detection = not show_detection
    elif key == ord('c'):  # Toggle controls window
        show_controls = not show_controls
        if show_controls:
            cv2.namedWindow('Controls')
        else:
            cv2.destroyWindow('Controls')
    
    # Show/hide controls window based on state
    if show_controls:
        # Create control panel with labels
        control_panel = np.ones((200, 400, 3), dtype=np.uint8) * 50
        
        # Add labels
        cv2.putText(control_panel, "Brightness: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        cv2.putText(control_panel, "Contrast: ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        cv2.putText(control_panel, "Detection Confidence: ", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        
        # Show current values
        cv2.putText(control_panel, f"{brightness}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        cv2.putText(control_panel, f"{contrast:.2f}", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        cv2.putText(control_panel, f"{detection_threshold:.2f}", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 1)
        
        cv2.imshow('Controls', control_panel)
    
    # Save frame if recording
    if recording:
        video_writer.write(frame)

video_capture.release()
cv2.destroyAllWindows()