import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pandas as pd
import pickle
from datetime import datetime
from pycaret.regression import load_model, predict_model

class RealtimeDepthPredictor:
    def __init__(self, current_location='thirdfloor'):
        # ROI settings - fixed 20x20 pixel region
        self.roi_point1 = None
        self.roi_point2 = None
        self.roi_selected = False
        
        # Image dimensions
        self.width = 640
        self.height = 480
        
        # Current location for prediction
        self.current_location = current_location
        print(f"Current location set to: {self.current_location}")
        
        # Try to load the model in different ways
        self.model_path = os.path.join("newEnvironmentTest", "Realsense", "model", "final_calibrated_depth_model.pkl")
        self.model = self.try_load_model()

        # Initialize manual calibration parameters as backup (simple linear adjustment)
        self.manual_calib_slope = 0.95  # Typical depth cameras overestimate distance by ~5%
        self.manual_calib_offset = 0.02 # Small fixed offset in meters
    
    def try_load_model(self):
        """Try multiple methods to load the calibration model"""
        try:
            # First try direct pickle loading
            try:
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"Successfully loaded model with pickle from: {self.model_path}")
                return model
            except Exception as pickle_error:
                print(f"Pickle loading failed: {pickle_error}")
            
            # Then try PyCaret loading
            try:
                model = load_model(self.model_path)
                print(f"Successfully loaded model with PyCaret from: {self.model_path}")
                return model
            except Exception as pycaret_error:
                print(f"PyCaret loading failed: {pycaret_error}")
            
            # If all fails, create a simple dummy model (just a function)
            print("Creating simple manual calibration function")
            def simple_calibration(depth):
                return depth * self.manual_calib_slope + self.manual_calib_offset
            
            return simple_calibration
            
        except Exception as e:
            print(f"All model loading methods failed: {e}")
            print("Will proceed with manual calibration")
            return None
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        # Start pipeline
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            print(f"Error starting camera: {e}")
            print("Please check that the camera is connected and not in use by another program")
            exit()
        
        # Get depth scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"Using RealSense Camera - Depth Scale: {self.depth_scale}")
        
        # Set up window and mouse callback
        cv2.namedWindow('RealSense Real-time Prediction')
        cv2.setMouseCallback('RealSense Real-time Prediction', self.mouse_callback)
        
        # Current measurements
        self.current_avg_depth = None
        self.calibrated_depth = None
        
        # Setup alignment
        self.align = rs.align(rs.stream.color)
        
        # Variables for running average
        self.running_avg_window = []
        self.window_size = 5  # Number of frames to average
        self.running_calibrated_window = []
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection - creates fixed 20x20 ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            norm_x, norm_y, is_right = self.normalize_coordinates(x, y)
            
            # Calculate top-left corner for a 20x20 ROI centered on the click point
            x1 = max(0, norm_x - 10)
            y1 = max(0, norm_y - 10)
            
            # Make sure the ROI stays within image boundaries
            if x1 + 20 >= self.width:
                x1 = self.width - 21
            if y1 + 20 >= self.height:
                y1 = self.height - 21
            
            # Set ROI points
            self.roi_point1 = (x1, y1)
            self.roi_point2 = (x1 + 20, y1 + 20)
            self.roi_selected = True
            
            # Show information about selected area
            print(f"Selected 20x20 ROI: {self.roi_point1} to {self.roi_point2}")
    
    def normalize_coordinates(self, x, y):
        """Convert coordinates from the combined image to single image coordinates"""
        normalized_x = x
        normalized_y = y
        is_right = False
        
        # Check if click is on the right (depth) image
        if x >= self.width:
            normalized_x = x - self.width
            is_right = True
        
        return normalized_x, normalized_y, is_right
    
    def get_normalized_roi(self):
        """Return normalized ROI coordinates (top-left, bottom-right)"""
        if not self.roi_selected or not self.roi_point1 or not self.roi_point2:
            return None
            
        x1, y1 = self.roi_point1
        x2, y2 = self.roi_point2
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, self.width-1))
        y1 = max(0, min(y1, self.height-1))
        x2 = max(0, min(x2, self.width-1))
        y2 = max(0, min(y2, self.height-1))
            
        return (x1, y1, x2, y2)
    
    def calculate_average_depth(self, depth_frame):
        """Calculate average depth within selected ROI"""
        roi = self.get_normalized_roi()
        if not roi:
            return None
            
        x1, y1, x2, y2 = roi
        
        try:
            # Use numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            roi_depth = depth_image[y1:y2+1, x1:x2+1]
            
            # Filter out zeros
            valid_depth = roi_depth[roi_depth > 0]
            
            if valid_depth.size == 0:
                return 0.0
            
            # Calculate average and convert to meters
            avg_depth = np.mean(valid_depth) * self.depth_scale
            return avg_depth
        except Exception as e:
            print(f"Error calculating depth: {e}")
            return None
    
    def predict_calibrated_depth(self, measured_depth):
        """Use the PyCaret model to predict calibrated depth"""
        if self.model is None or measured_depth is None or measured_depth <= 0:
            return None
        
        try:
            # Create input DataFrame with measured depth and location
            input_data = pd.DataFrame({
                'average_depth_m': [measured_depth],
                'Location': [self.current_location]
            })
            
            # Make prediction
            prediction = predict_model(self.model, data=input_data, verbose=False)
            
            # Extract prediction value
            calibrated_depth = prediction['prediction_label'].iloc[0]
            return calibrated_depth
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def update_running_average(self, value, window):
        """Update running average for smoother readings"""
        if value is None:
            return None
        
        window.append(value)
        if len(window) > self.window_size:
            window.pop(0)
        
        return sum(window) / len(window)
    
    def run(self):
        """Main loop for real-time depth prediction"""
        try:
            print("Started program - Click anywhere to place a 20x20 pixel ROI for depth measurement")
            print("Press 'l' to toggle location between 'thirdfloor' and 'outdoor1stfloor'")
            print("Press 'q' to exit the program")
            print(f"Current location: {self.current_location}")
            
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert frames to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Create colorized depth map
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Create copies for drawing
                color_display = color_image.copy()
                depth_display = depth_colormap.copy()
                
                # Calculate and predict depth if ROI is selected
                if self.roi_selected:
                    # Calculate raw average depth
                    measured_depth = self.calculate_average_depth(depth_frame)
                    
                    # Update running average for measured depth
                    self.current_avg_depth = self.update_running_average(
                        measured_depth, 
                        self.running_avg_window
                    )
                    
                    # Predict calibrated depth if model is available
                    if self.current_avg_depth is not None and self.current_avg_depth > 0 and self.model is not None:
                        calibrated_depth = self.predict_calibrated_depth(self.current_avg_depth)
                        
                        # Update running average for calibrated depth
                        self.calibrated_depth = self.update_running_average(
                            calibrated_depth,
                            self.running_calibrated_window
                        )
                
                # Display ROI if there's one
                if self.roi_selected:
                    roi = self.get_normalized_roi()
                    if roi:
                        x1, y1, x2, y2 = roi
                        
                        # Draw rectangle on both images
                        cv2.rectangle(color_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(depth_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Display depth values
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        if self.current_avg_depth is not None:
                            measured_text = f"Measured : {self.current_avg_depth:.3f} m"
                            cv2.putText(color_display, measured_text, (10, 60), 
                                      font, font_scale, (0, 255, 255), thickness)
                        
                        if self.calibrated_depth is not None:
                            # Get calibration type message
                            if callable(self.model) and not hasattr(self.model, 'predict'):
                                cal_type = "(Manual Calibration)"
                            else:
                                cal_type = "(Model-based)"
                                
                            calibrated_text = f"Calibrated: {self.calibrated_depth:.3f} m {cal_type}"
                            cv2.putText(color_display, calibrated_text, (10, 90), 
                                      font, font_scale, (0, 255, 0), thickness)
                
                # Combine color and depth images
                images = np.hstack((color_display, depth_display))
                
                # Add dividing line between images
                cv2.line(images, (self.width, 0), (self.width, self.height), (255, 255, 255), 1)
                
                # Add labels and info
                cv2.putText(images, "Color", (10, 20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(images, "Depth", (self.width + 10, 20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(images, f"Location: {self.current_location}", (self.width - 250, 20), 
                           font, 0.5, (255, 255, 255), 1)
                
                # Display error correction notification if using fallback
                if self.model is None:
                    cv2.putText(images, "Using fallback correction (no model)", (10, 120), 
                               font, 0.6, (0, 0, 255), 2)
                
                # Display timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(images, timestamp, (10, images.shape[0] - 20), 
                           font, 0.5, (255, 255, 255), 1)
                
                # Display combined image
                cv2.imshow('RealSense Real-time Prediction', images)
                
                # Check for keypresses
                key = cv2.waitKey(1) & 0xFF
                
                # Press 'q' to exit
                if key == ord('q'):
                    break
                    
    
                
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Clean up
            print("Shutting down camera...")
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Program terminated")

if __name__ == "__main__":
    predictor = RealtimeDepthPredictor
    predictor.run()