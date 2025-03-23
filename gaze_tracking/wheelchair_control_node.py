#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
from gaze_tracking import GazeTracking

# Wheelchair control states
class WheelchairState:
    STOP = "STOP"
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class WheelchairControlNode(Node):
    def __init__(self):
        super().__init__('wheelchair_control_node')
        
        # Initialize publishers
        self.wheelchair_state_pub = self.create_publisher(
            String,
            'wheelchair_state',
            10)
        
        # Initialize gaze tracking and webcam
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(2)
        
        # Wheelchair control variables
        self.wheelchair_state = WheelchairState.STOP
        self.last_state_change_time = self.get_clock().now()
        self.eye_closed_timeout = 3.0  # Stop wheelchair if eyes are closed for 3 seconds
        
        # Buffer for smoothing gaze direction
        self.gaze_buffer_size = 5
        self.gaze_buffer = []
        
        # Calibration data
        self.calibration_data = {
            "center_gaze_ratio": 0.5,
            "left_threshold": 0.6,
            "right_threshold": 0.4,
            "blinking_threshold": 3.8,
        }
        
        # Create timer for periodic publishing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz rate
        
        self.get_logger().info('Wheelchair Control Node has been initialized')
        
    def calibrate(self):
        """Perform calibration of gaze tracking"""
        self.get_logger().info('Starting calibration...')
        
        def collect_gaze_data(prompt, duration):
            self.get_logger().info(prompt)
            # Wait for 2 seconds using ROS2 time
            start_time = self.get_clock().now()
            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < 2.0:
                rclpy.spin_once(self)
            
            gaze_ratios = []
            blinking_ratios = []
            
            start_time = self.get_clock().now()
            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
                _, frame = self.webcam.read()
                self.gaze.refresh(frame)
                
                if self.gaze.pupils_located:
                    gaze_ratio = self.gaze.horizontal_ratio()
                    gaze_ratios.append(gaze_ratio)
                    blinking_ratios.append((self.gaze.eye_left.blinking + self.gaze.eye_right.blinking) / 2)
                
                # Small delay between frames
                rclpy.spin_once(self)
            
            if gaze_ratios:
                return np.mean(gaze_ratios), np.mean(blinking_ratios)
            return None, None
        
        # Perform calibration steps
        center_gaze_ratio, _ = collect_gaze_data("Please look at the center of the screen (forward).", 5)
        if center_gaze_ratio is not None:
            self.calibration_data["center_gaze_ratio"] = center_gaze_ratio
            self.get_logger().info(f'Center gaze ratio: {center_gaze_ratio}')
        
        left_gaze_ratio, _ = collect_gaze_data("Please look to the left.", 5)
        if left_gaze_ratio is not None:
            self.calibration_data["left_threshold"] = (center_gaze_ratio + left_gaze_ratio) / 2
            self.get_logger().info(f'Left gaze ratio: {left_gaze_ratio}')
        
        right_gaze_ratio, _ = collect_gaze_data("Please look to the right.", 5)
        if right_gaze_ratio is not None:
            self.calibration_data["right_threshold"] = (center_gaze_ratio + right_gaze_ratio) / 2
            self.get_logger().info(f'Right gaze ratio: {right_gaze_ratio}')
        
        _, blinking_ratio = collect_gaze_data("Please blink your eyes.", 5)
        if blinking_ratio is not None:
            self.calibration_data["blinking_threshold"] = blinking_ratio * 1.2
            self.get_logger().info(f'Blinking ratio: {blinking_ratio}')
        
        self.get_logger().info('Calibration complete!')
    
    def timer_callback(self):
        """Periodic callback to process gaze tracking and publish wheelchair state"""
        # Get a new frame from the webcam
        _, frame = self.webcam.read()
        
        # Analyze the frame
        self.gaze.refresh(frame)
        
        # Get wheelchair state
        wheelchair_state = String()
        
        if self.gaze.pupils_located:
            gaze_ratio = self.gaze.horizontal_ratio()
            self.gaze_buffer.append(gaze_ratio)
            if len(self.gaze_buffer) > self.gaze_buffer_size:
                self.gaze_buffer.pop(0)
            
            # Smooth the gaze ratio using moving average
            smoothed_gaze_ratio = np.mean(self.gaze_buffer)
            
            # Determine gaze direction and wheelchair state
            blinking_ratio = (self.gaze.eye_left.blinking + self.gaze.eye_right.blinking) / 2
            current_time = self.get_clock().now()
            
            if blinking_ratio > self.calibration_data["blinking_threshold"]:
                wheelchair_state.data = WheelchairState.STOP
                if (current_time - self.last_state_change_time).nanoseconds / 1e9 > self.eye_closed_timeout:
                    self.wheelchair_state = WheelchairState.STOP
                    self.last_state_change_time = current_time
            elif smoothed_gaze_ratio <= self.calibration_data["right_threshold"]:
                wheelchair_state.data = WheelchairState.RIGHT
                self.wheelchair_state = WheelchairState.RIGHT
                self.last_state_change_time = current_time
            elif smoothed_gaze_ratio >= self.calibration_data["left_threshold"]:
                wheelchair_state.data = WheelchairState.LEFT
                self.wheelchair_state = WheelchairState.LEFT
                self.last_state_change_time = current_time
            else:
                wheelchair_state.data = WheelchairState.FORWARD
                self.wheelchair_state = WheelchairState.FORWARD
                self.last_state_change_time = current_time
        else:
            wheelchair_state.data = WheelchairState.STOP
            self.wheelchair_state = WheelchairState.STOP
        
        # Publish the wheelchair state
        self.wheelchair_state_pub.publish(wheelchair_state)
    
    def __del__(self):
        """Cleanup when the node is destroyed"""
        self.webcam.release()

def main(args=None):
    rclpy.init(args=args)
    node = WheelchairControlNode()
    
    try:
        # Run calibration first
        node.calibrate()
        
        # Run the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 