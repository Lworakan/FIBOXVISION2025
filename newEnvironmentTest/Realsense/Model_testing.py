#!/usr/bin/env python3
"""
Real-time depth prediction with RealSense camera using a Region of Interest (ROI)
for more accurate depth measurements with aligned color frames.
"""

import os
import numpy as np
import pandas as pd
import cv2
import time
import pyrealsense2 as rs
from pycaret.regression import load_model, predict_model
from collections import deque

MODEL_PATH = '/home/lworakan/Documents/GitHub/FIBOXVISION2025/newEnvironmentTest/Realsense/model/final_calibrated_depth_model_outdoor'

class RunningAverage:
    def __init__(self, window_size=30):
        self.values = deque(maxlen=window_size)
        
    def update(self, new_value):
        if new_value is not None and not np.isnan(new_value):
            self.values.append(new_value)
            
    def get_average(self):
        if not self.values:
            return None
        return np.mean(self.values)
    
    def get_std(self):
        if len(self.values) < 2:
            return 0
        return np.std(self.values)
    
    def clear(self):
        self.values.clear()

def main():
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Initialize RealSense camera
        print("Initializing RealSense camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Create alignment object
        align = rs.align(rs.stream.color)
        
        # Start streaming
        profile = pipeline.start(config)
        print("Camera started. Warming up for 2 seconds...")
        time.sleep(2)
        
        # Get depth sensor and set options if needed
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # Enable auto-exposure for depth
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        
        # Variables for tracking
        location = "Lab"  # Default location
        intended_distance = None
        saved_predictions = []
        
        raw_depth_avg = RunningAverage(30)  
        calibrated_depth_avg = RunningAverage(15)  
        roi_size = 20
        roi_x = (640 - roi_size) // 2  # Center X
        roi_y = (480 - roi_size) // 2  # Center Y
        roi_dragging = False
        
        # Display options
        show_color = True  
        print("\nControls:")
        print("  'q' - Quit the application")
        print("  's' - Save current prediction to results")
        print("  'd' - Set intended/actual distance")
        print("  'l' - Change location")
        print("  'v' - Toggle between color and depth view")
        print("  Mouse - Click and drag to move the ROI")
        
        # Mouse callback function for ROI selection
        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_x, roi_y, roi_dragging
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is inside ROI
                if (roi_x <= x <= roi_x + roi_size) and (roi_y <= y <= roi_y + roi_size):
                    roi_dragging = True
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if roi_dragging:
                    roi_x = max(0, min(x - roi_size//2, 640 - roi_size))
                    roi_y = max(0, min(y - roi_size//2, 480 - roi_size))
            
            elif event == cv2.EVENT_LBUTTONUP:
                roi_dragging = False
        
        cv2.namedWindow('RealSense Camera')
        cv2.setMouseCallback('RealSense Camera', mouse_callback)
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            color_display = color_image.copy()
            depth_display = depth_colormap.copy()
            
            roi = depth_image[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
            
            roi_depth_values = roi[roi > 0].astype(float) / 1000.0  # convert to meters
            
            if len(roi_depth_values) == 0:
                average_depth = 0
            else:
                sample_size = 20
                if len(roi_depth_values) > sample_size:
                    roi_depth_values = roi_depth_values[-sample_size:]
                
                average_depth = np.mean(roi_depth_values)
                
                raw_depth_avg.update(average_depth)
                
                stabilized_depth = raw_depth_avg.get_average()
                
                if stabilized_depth is not None and stabilized_depth > 0:
                    data = pd.DataFrame({
                        'average_depth_m': [stabilized_depth],
                        'Location': [location]
                    })
                    
                    prediction = predict_model(model, data=data)
                    predicted_distance = prediction['prediction_label'].iloc[0]
                    
                    calibrated_depth_avg.update(predicted_distance)
                    calibrated_distance = calibrated_depth_avg.get_average()
                    
                    error_text = ""
                    if intended_distance is not None:
                        error = abs(calibrated_distance - intended_distance)
                        error_percent = (error / intended_distance) * 100 if intended_distance > 0 else 0
                        error_text = f"Error: {error:.3f}m ({error_percent:.1f}%)"
                    
                    for display in [color_display, depth_display]:
                        cv2.rectangle(
                            display,
                            (roi_x, roi_y),
                            (roi_x + roi_size, roi_y + roi_size),
                            (255, 255, 255) if display is depth_display else (255, 255, 255),
                            2
                        )
                        
                        info_text = [
                            f"Location: {location}",
                            f"Raw Depth: {stabilized_depth:.3f}m",
                            f"Calibrated: {calibrated_distance:.3f}m",
                        ]
                        
                        if intended_distance is not None:
                            info_text.append(f"Actual: {intended_distance:.3f}m")
                            info_text.append(error_text)
                        
                        for i, text in enumerate(info_text):
                            cv2.putText(
                                display, 
                                text, 
                                (20, 30 + 30*i), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (255, 255, 255) if display is depth_display else (255, 255, 255), 
                                2
                            )
                    
                    # Print to console
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"Location: {location}")
                    print(f"Raw Depth (stabilized): {stabilized_depth:.4f}m")
                    print(f"Calibrated Distance: {calibrated_distance:.4f}m ({calibrated_distance*100:.1f}cm)")
                    
                    if intended_distance is not None:
                        print(f"Actual Distance: {intended_distance:.4f}m")
                        print(f"Absolute Error: {error:.4f}m ({error*100:.1f}cm)")
                        print(f"Percentage Error: {error_percent:.2f}%")
                    else:
                        for display in [color_display, depth_display]:
                            # Draw ROI rectangle (red to indicate no valid data)
                            cv2.rectangle(
                                display,
                                (roi_x, roi_y),
                                (roi_x + roi_size, roi_y + roi_size),
                                (255, 255, 255) if display is depth_display else (255, 255, 255),
                                2
                            )
                   
                    # cv2.putText(
                    #     display,
                    #     "No valid depth data in ROI",
                    #     (20, 30),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.7,
                    #     (0, 0, 255) if display is depth_display else (255, 255, 255),
                    #     2
                    # )
            
            display_image = depth_display if show_color else color_display
            cv2.imshow('RealSense Camera', display_image)
            
            key = cv2.waitKey(1)
            
            # Quit on 'q'
            if key & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # Toggle view on 'v'
            elif key & 0xFF == ord('v'):
                show_color = not show_color
                print(f"\nSwitched to {'color' if show_color else 'depth'} view")
            
            # Save prediction on 's'
            elif key & 0xFF == ord('s') and 'calibrated_distance' in locals():
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                data = {
                    'timestamp': timestamp,
                    'raw_depth_m': stabilized_depth,
                    'calibrated_distance_m': calibrated_distance,
                    'location': location,
                }
                
                if intended_distance is not None:
                    data['actual_distance_m'] = intended_distance
                    data['error_m'] = error
                    data['error_percent'] = error_percent
                
                saved_predictions.append(data)
                print(f"\nSaved prediction at {timestamp}")
                print(f"Total saved: {len(saved_predictions)}")
                time.sleep(0.5)  # Brief pause to show confirmation
            
            elif key & 0xFF == ord('d'):
                try:
                    input_value = input("\nEnter actual/intended distance in meters: ")
                    intended_distance = float(input_value)
                    print(f"Actual distance set to {intended_distance:.3f}m")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            # Change location on 'l'
            elif key & 0xFF == ord('l'):
                location = input("\nEnter location name: ")
                print(f"Location set to: {location}")
                raw_depth_avg.clear()  # Clear running averages when location changes
                calibrated_depth_avg.clear()
            
            # Change ROI size with + and -
            elif key & 0xFF == ord('+') or key & 0xFF == ord('='):
                roi_size = min(200, roi_size + 10)
                # Adjust ROI position to keep it within bounds
                roi_x = min(roi_x, 640 - roi_size)
                roi_y = min(roi_y, 480 - roi_size)
                print(f"ROI size increased to {roi_size}x{roi_size} pixels")
            
            elif key & 0xFF == ord('-') or key & 0xFF == ord('_'):
                roi_size = max(20, roi_size - 10)
                print(f"ROI size decreased to {roi_size}x{roi_size} pixels")
            
            # Limit update rate
            time.sleep(0.03)
        
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # Save results if any were collected
        if saved_predictions:
            results_df = pd.DataFrame(saved_predictions)
            filename = f"depth_predictions_{time.strftime('%Y%m%d-%H%M%S')}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nSaved {len(saved_predictions)} predictions to {filename}")
            
            # Display statistics if we have actual distances
            if 'error_m' in results_df.columns:
                print("\nError Statistics:")
                print(f"Mean Error: {results_df['error_m'].mean():.4f}m")
                print(f"RMSE: {np.sqrt((results_df['error_m']**2).mean()):.4f}m")
                print(f"Mean Error %: {results_df['error_percent'].mean():.2f}%")
                
                # Group by location if we have multiple locations
                if results_df['location'].nunique() > 1:
                    print("\nError Statistics by Location:")
                    for loc in results_df['location'].unique():
                        loc_df = results_df[results_df['location'] == loc]
                        print(f"\n{loc}:")
                        print(f"  Mean Error: {loc_df['error_m'].mean():.4f}m")
                        print(f"  RMSE: {np.sqrt((loc_df['error_m']**2).mean()):.4f}m")
                        print(f"  Number of samples: {len(loc_df)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()