import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        """Initialize MediaPipe pose detection components"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_landmarks(self, frame):
        """
        Detect pose landmarks in a frame
        
        Args:
            frame (np.array): Input video frame
            
        Returns:
            landmarks (dict): Detected pose landmarks if found, None otherwise
            annotated_frame (np.array): Frame with pose landmarks drawn
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for pose detection
        results = self.pose.process(frame_rgb)
        
        # Draw landmarks on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            return results.pose_landmarks, annotated_frame
        
        return None, annotated_frame
    
    def calculate_arm_angle(self, landmarks):
        """
        Calculate the shooting arm angle
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Arm angle in degrees
        """
        # Extract relevant landmarks for right arm (assuming right-handed shooter)
        # If landmarks are not available, return None
        if not landmarks:
            return None
            
        # Get shoulder, elbow, and wrist points
        shoulder = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                           landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        elbow = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                         landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y])
        wrist = np.array([landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                         landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y])
        
        # Calculate vectors
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        
        # Calculate angle using dot product
        dot_product = np.dot(upper_arm, forearm)
        norm_product = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
        
        if norm_product == 0:
            return None
            
        cos_angle = dot_product / norm_product
        # Clip to avoid domain errors from floating point imprecision
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg