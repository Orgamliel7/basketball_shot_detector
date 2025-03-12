import numpy as np

class ShotAnalyzer:
    def __init__(self):
        """Initialize shot analysis parameters"""
        # Frame history to track motion
        self.arm_angle_history = []
        self.max_history_length = 30  # ~1 second at 30fps
        
        # Shot detection thresholds
        self.shot_angle_threshold = 160  # Minimum angle for shot detection
        self.angle_change_threshold = 60  # Minimum change in angle to detect shooting motion
        
        # State tracking
        self.is_shooting = False
        self.shot_start_time = None
        self.cooldown_frames = 0
        self.cooldown_period = 45  # ~1.5 seconds cooldown after detecting a shot
    
    def update(self, arm_angle, frame_count):
        """
        Update the shot analyzer with new arm angle data
        
        Args:
            arm_angle (float): Current arm angle
            frame_count (int): Current frame number
            
        Returns:
            dict: Shot information if a shot is detected, None otherwise
        """
        # Skip processing if in cooldown period
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return None
            
        # If no valid arm angle, just update history
        if arm_angle is None:
            self.arm_angle_history.append(0)  # Use zero as placeholder
        else:
            self.arm_angle_history.append(arm_angle)
            
        # Keep history at fixed length
        if len(self.arm_angle_history) > self.max_history_length:
            self.arm_angle_history.pop(0)
            
        # Need sufficient history to detect shots
        if len(self.arm_angle_history) < 10:
            return None
            
        # Check for shot motion pattern
        return self._detect_shot_motion(frame_count)
    
    def _detect_shot_motion(self, frame_count):
        """
        Detect basketball shot based on arm motion pattern
        
        Args:
            frame_count (int): Current frame number
            
        Returns:
            dict: Shot information or None
        """
        # Get recent history without None values
        valid_angles = [a for a in self.arm_angle_history[-10:] if a > 0]
        
        if len(valid_angles) < 5:
            return None
            
        # Check for the distinctive pattern:
        # 1. Arm extended (large angle)
        # 2. Followed by a significant decrease (shooting motion)
        
        max_angle = max(valid_angles)
        min_recent_angle = min(valid_angles[-3:])  # Look at most recent angles
        angle_change = max_angle - min_recent_angle
        
        # Basic shooting detection logic
        if not self.is_shooting and max_angle > self.shot_angle_threshold and angle_change > self.angle_change_threshold:
            self.is_shooting = True
            self.shot_start_time = frame_count
            return None  # Still collecting data about the shot
            
        # If we're already tracking a shot, check if it's complete
        elif self.is_shooting:
            # Shot is considered complete when the arm angle starts increasing again
            if len(valid_angles) >= 3 and valid_angles[-1] > valid_angles[-2]:
                shot_info = {
                    "detected": True,
                    "frame": frame_count,
                    "max_arm_angle": max_angle,
                    "shot_type": self._classify_shot_type(max_angle, angle_change)
                }
                
                # Reset state and set cooldown
                self.is_shooting = False
                self.cooldown_frames = self.cooldown_period
                self.arm_angle_history = []
                
                return shot_info
                
        return None
    
    def _classify_shot_type(self, max_angle, angle_change):
        """
        Classify the type of shot based on arm angle characteristics
        
        Args:
            max_angle (float): Maximum arm angle during shot
            angle_change (float): Change in arm angle during shot
            
        Returns:
            str: Shot type classification
        """
        # Very simple classification based on arm extension
        if max_angle > 170:
            return "Free Throw"
        elif max_angle > 150:
            return "Jump Shot"
        else:
            return "Quick Shot"