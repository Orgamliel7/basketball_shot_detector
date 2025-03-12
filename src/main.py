import cv2
import numpy as np
import time
from pose_detector import PoseDetector
from shot_analyzer import ShotAnalyzer

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
    
    # Tracking variables
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Last shot information
    last_shot_info = None
    shot_display_frames = 0
    
    print("Basketball Shot Detector started. Press 'q' to quit.")
    
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
        
        # Display the processed frame
        cv2.imshow("Basketball Shot Detector", annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()