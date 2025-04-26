import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import time
import datetime
import os

# Pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale is: {depth_scale}")

# Recording setup
recording = False
color_writer = None
depth_writer = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for .mp4
fps = 30
color_filename = 'color_output.mp4'
depth_filename = 'depth_output.mp4'

# CSV setup
csv_filename = 'depth_data.csv'
csv_file = None
csv_writer = None
CSV_HEADER = ['Timestamp', 'Frame', 'Center_X', 'Center_Y', 'Center_Depth_m', 
              'Min_Depth_m', 'Max_Depth_m', 'Avg_Depth_m', 'Std_Dev_m']
frame_count = 0

def get_depth_stats(depth_frame, x_center, y_center, radius=10):
    """Get depth statistics from a region around the center point"""
    height, width = depth_frame.shape
    
    # Define region of interest
    x_min = max(0, x_center - radius)
    x_max = min(width - 1, x_center + radius)
    y_min = max(0, y_center - radius)
    y_max = min(height - 1, y_center + radius)
    
    # Extract ROI
    roi = depth_frame[y_min:y_max, x_min:x_max]
    
    # Filter out zero depths (no valid measurement)
    valid_depths = roi[roi > 0]
    
    if valid_depths.size > 0:
        # Convert to meters
        valid_depths_m = valid_depths * depth_scale
        
        # Calculate statistics
        center_depth_m = depth_frame[y_center, x_center] * depth_scale if depth_frame[y_center, x_center] > 0 else 0
        min_depth_m = np.min(valid_depths_m)
        max_depth_m = np.max(valid_depths_m)
        avg_depth_m = np.mean(valid_depths_m)
        std_dev_m = np.std(valid_depths_m)
        
        return center_depth_m, min_depth_m, max_depth_m, avg_depth_m, std_dev_m
    else:
        return 0, 0, 0, 0, 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Get frame data as numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get image dimensions
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2

        # Convert and colorize depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Flip vertically for both images
        color_image = cv2.flip(color_image, 0)
        depth_colormap = cv2.flip(depth_colormap, 0)
        depth_image_flipped = cv2.flip(depth_image, 0)
        
        # Update frame counter
        frame_count += 1

        # Add crosshair at center for reference
        cv2.line(color_image, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
        cv2.line(color_image, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(depth_colormap, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
        cv2.line(depth_colormap, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        
        # Get depth statistics
        center_depth_m, min_depth_m, max_depth_m, avg_depth_m, std_dev_m = get_depth_stats(
            depth_image_flipped, center_x, center_y)
        
        # Display depth info
        depth_text = f"Depth: {center_depth_m:.3f}m"
        cv2.putText(color_image, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(depth_colormap, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Start recording if toggled on
        if recording:
            # Initialize video writers if not already done
            if color_writer is None or depth_writer is None:
                h, w, _ = color_image.shape
                color_writer = cv2.VideoWriter(color_filename, fourcc, fps, (w, h))
                depth_writer = cv2.VideoWriter(depth_filename, fourcc, fps, (w, h))
            
            # Write frames to video
            color_writer.write(color_image)
            depth_writer.write(depth_colormap)
            
            # Initialize CSV writer if not already done
            if csv_file is None:
                file_exists = os.path.exists(csv_filename)
                csv_file = open(csv_filename, 'a', newline='')
                csv_writer = csv.writer(csv_file)
                if not file_exists:
                    csv_writer.writerow(CSV_HEADER)
                    print(f"Created new CSV file: {csv_filename}")
                else:
                    print(f"Appending to existing CSV file: {csv_filename}")
            
            # Write depth data to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            csv_writer.writerow([
                timestamp, 
                frame_count, 
                center_x, 
                center_y, 
                f"{center_depth_m:.6f}", 
                f"{min_depth_m:.6f}", 
                f"{max_depth_m:.6f}", 
                f"{avg_depth_m:.6f}", 
                f"{std_dev_m:.6f}"
            ])
            
            # Indicate recording status
            cv2.putText(color_image, "REC", (width - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            cv2.putText(depth_colormap, "REC", (width - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)

        # Stack for visualization
        stacked = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense RGB + Depth', stacked)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            recording = not recording
            print("Recording started." if recording else "Recording stopped.")
            if not recording:
                if color_writer:
                    color_writer.release()
                    color_writer = None
                if depth_writer:
                    depth_writer.release()
                    depth_writer = None
                if csv_file:
                    csv_file.flush()
                    csv_file.close()
                    csv_file = None
                    print(f"CSV data saved to {csv_filename}")
        elif key == ord('s') and recording:
            # Manual save/flush for CSV without stopping recording
            if csv_file:
                csv_file.flush()
                print(f"CSV data flushed to {csv_filename}")

finally:
    pipeline.stop()
    if color_writer:
        color_writer.release()
    if depth_writer:
        depth_writer.release()
    if csv_file and not csv_file.closed:
        csv_file.close()
        print(f"CSV data saved to {csv_filename}")
    cv2.destroyAllWindows()
    print("Application closed.")