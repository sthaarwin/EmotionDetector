import cv2
import numpy as np
import time
from datetime import datetime
from fer import FER

# Initialize detectors
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Change FER initialization to use default backend
emotion_detector = FER(mtcnn=False)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print('Unable to load camera.')
    exit()

# Initialize variables
show_detection = True
prev_frame_time = 0
new_frame_time = 0

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
    
    try:
        # Detect emotions with error handling
        emotions = emotion_detector.detect_emotions(frame)
        
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

    cv2.imshow('Video', frame)
    
    # Handle keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Save screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f'screenshot_{timestamp}.jpg', frame)
    elif key == ord('t'):  # Toggle detection boxes
        show_detection = not show_detection

video_capture.release()
cv2.destroyAllWindows()