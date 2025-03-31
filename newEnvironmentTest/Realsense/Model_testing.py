#!/usr/bin/env python3
"""
Real-time depth prediction with RealSense camera using a Region of Interest (ROI)
for more accurate depth measurements.
"""

import os
import numpy as np
import pandas as pd
import cv2
import time
import pyrealsense2 as rs
from pycaret.regression import load_model, predict_model

# Use the exact path that works
MODEL_PATH = '/home/lworakan/Documents/GitHub/FIBOXVISION2025/newEnvironmentTest/Realsense/model/final_calibrated_depth_model'

def main():
    try:
        # Load the model (without adding .pkl)
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Initialize RealSense camera
        print("Initializing RealSense camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        print("Camera started. Warming up for 2 seconds...")
        time.sleep(2)
        
        # Get stream profile and camera intrinsics
        profile = pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        
        # Variables for tracking
        location = "Lab"  # Default location
        intended_distance = None
        saved_predictions = []
        
        # ROI parameters (initially center of the frame)
        roi_size = 20  # Size of the ROI square in pixels
        roi_x = (640 - roi_size) // 2  # Center X
        roi_y = (480 - roi_size) // 2  # Center Y
        roi_dragging = False
        
        # Display instructions
        print("\nControls:")
        print("  'q' - Quit the application")
        print("  's' - Save current prediction to results")
        print("  'd' - Set intended/actual distance")
        print("  'l' - Change location")
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
                    # Update ROI position, ensuring it stays within frame
                    roi_x = max(0, min(x - roi_size//2, 640 - roi_size))
                    roi_y = max(0, min(y - roi_size//2, 480 - roi_size))
            
            elif event == cv2.EVENT_LBUTTONUP:
                roi_dragging = False
        
        # Create named window and set mouse callback
        cv2.namedWindow('Depth Prediction')
        cv2.setMouseCallback('Depth Prediction', mouse_callback)
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Create ROI mask
            roi = depth_image[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
            
            # Calculate average depth in ROI (ignoring zeros)
            roi_depth_values = roi[roi > 0].astype(float) / 1000.0  # convert to meters
            
            if len(roi_depth_values) == 0:
                average_depth = 0
            else:
                average_depth = np.mean(roi_depth_values)
            
            # Apply colormap to the depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Draw ROI on the depth colormap
            cv2.rectangle(
                depth_colormap,
                (roi_x, roi_y),
                (roi_x + roi_size, roi_y + roi_size),
                (255, 255, 255),
                2
            )
            
            if average_depth > 0:
                ## mean average_depth 100 data
                sample_size = 100
                if len(roi_depth_values) > sample_size:
                    roi_depth_values = roi_depth_values[-sample_size:]
                average_depth = np.mean(roi_depth_values)
                data = pd.DataFrame({
                    'average_depth_m': [average_depth],
                    'Location': [location]
                })
                
                # Make prediction
                prediction = predict_model(model, data=data)
                predicted_distance = prediction['prediction_label'].iloc[0]
                
                # Calculate error if intended distance is set
                error_text = ""
                if intended_distance is not None:
                    error = abs(predicted_distance - intended_distance)
                    error_percent = (error / intended_distance) * 100 if intended_distance > 0 else 0
                    error_text = f"Error: {error:.3f}m ({error_percent:.1f}%)"
                
                # Add text to image
                info_text = [
                    f"Location: {location}",
                    f"ROI Avg Depth: {average_depth:.3f}m",
                    f"Predicted Distance: {predicted_distance:.3f}m",
                ]
                
                if intended_distance is not None:
                    info_text.append(f"Actual Distance: {intended_distance:.3f}m")
                    info_text.append(error_text)
                
                # Add text to the image
                for i, text in enumerate(info_text):
                    cv2.putText(
                        depth_colormap, 
                        text, 
                        (20, 30 + 30*i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )
                
                # Print to console as well
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Location: {location}")
                print(f"ROI Average Depth: {average_depth:.4f}m")
                print(f"Predicted Distance: {predicted_distance:.4f}m ({predicted_distance*100:.1f}cm)")
                
                if intended_distance is not None:
                    print(f"Actual Distance: {intended_distance:.4f}m")
                    print(f"Absolute Error: {error:.4f}m ({error*100:.1f}cm)")
                    print(f"Percentage Error: {error_percent:.2f}%")
            else:
                # No valid depth data in ROI
                cv2.putText(
                    depth_colormap,
                    "No valid depth data in ROI",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Display the colormap and ROI
            cv2.imshow('Depth Prediction', depth_colormap)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            # Quit on 'q'
            if key & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # Save prediction on 's'
            elif key & 0xFF == ord('s') and average_depth > 0:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                data = {
                    'timestamp': timestamp,
                    'depth_m': average_depth,
                    'predicted_distance_m': predicted_distance,
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
            
            # Set intended distance on 'd'
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
            time.sleep(0.05)
        
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