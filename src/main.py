import cv2
import numpy as np
import time
import sys
from src.pose_detector import PoseDetector
from src.shot_analyzer import ShotAnalyzer

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera properties (if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize our components
    pose_detector = PoseDetector()
    shot_analyzer = ShotAnalyzer()
    
    # Initialize face and smile detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Tracking variables
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Last shot information
    last_shot_info = None
    shot_display_frames = 0
    
    # Smile detection variables
    smile_detected = False
    smile_display_frames = 0
    
    print("Basketball Shot Detector started. Press 'q' to quit.")
    
    # Create named window with normal properties (to allow X button to work properly)
    cv2.namedWindow("Basketball Shot Detector", cv2.WINDOW_NORMAL)
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break
            
        # Update frame count
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Process frame with pose detector
        landmarks, annotated_frame = pose_detector.detect_landmarks(frame)
        
        # Calculate arm angle
        arm_angle = pose_detector.calculate_arm_angle(landmarks)
        
        # Update shot analyzer
        shot_info = shot_analyzer.update(arm_angle, frame_count)
        
        # If a shot is detected, store the information
        if shot_info:
            last_shot_info = shot_info
            shot_display_frames = 45  # Display for ~1.5 seconds
        
        # Convert to grayscale for face/smile detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each face for smile detection
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Region of interest for the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = annotated_frame[y:y + h, x:x + w]
            
            # Detect smiles
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            
            # If smile detected
            if len(smiles) > 0:
                smile_detected = True
                smile_display_frames = 45  # Display for ~1.5 seconds
                
                # Draw rectangle around smile
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
        
        # Display information on the frame
        # FPS counter
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # Arm angle if available
        if arm_angle:
            cv2.putText(annotated_frame, f"Arm Angle: {arm_angle:.1f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show shot detection result if available
        if last_shot_info and shot_display_frames > 0:
            cv2.putText(annotated_frame, f"Shot Detected: {last_shot_info['shot_type']}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            shot_display_frames -= 1
            
        # Show smile detection message if available
        if smile_detected and smile_display_frames > 0:
            cv2.putText(annotated_frame, "Nice smile! :)", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            smile_display_frames -= 1
            if smile_display_frames == 0:
                smile_detected = False
        
        # Display the processed frame
        cv2.imshow("Basketball Shot Detector", annotated_frame)
        
        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Basketball Shot Detector", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Ensure a clean exit
    print("Application closed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()