import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
import datetime
import csv
import os
import math

# Try to import RealSense library, but continue if not available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("RealSense SDK found")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("RealSense SDK not found")

# --- Configuration ---
MODEL_PATH = r"./Callback/yolo11l.pt"  # model path
OUTPUT_CSV_FILE = 'camera_tracking.csv'  # output csv file
# -----------------------------

# YOLO Model Configuration
CONF_THRESHOLD = 0.60  # Confidence threshold
TARGET_CLASS_NAME = 'basketball_hoop'  # Change this to your target class

# Camera Configuration
CAMERA_ID = 0  # Default webcam (change if you have multiple cameras)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# CSV Header
CSV_HEADER = ['Timestamp', 'Frame', 'X_min', 'Y_min', 'X_max', 'Y_max', 'Confidence', 
              'area', 'X', 'Y', 'Z_est', 'Distance_3D', 'Distance_RealSense']

# Reference object size in meters (used for Z estimation)
# This should be calibrated for your specific object
REFERENCE_OBJECT_WIDTH = 0.5  # meters
REFERENCE_OBJECT_HEIGHT = 0.5  # meters

# Focal length estimation (can be calibrated)
FOCAL_LENGTH = 500  # placeholder value

def main():
    # --- Ask user for camera choice ---
    print("\n===== Camera Selection =====")
    print("A: Use RealSense camera" if REALSENSE_AVAILABLE else "A: RealSense not available")
    print("B: Use notebook webcam")
    print("=========================")
    
    while True:
        choice = input("Enter your choice (A/B): ").strip().upper()
        if choice == 'A' and REALSENSE_AVAILABLE:
            USE_REALSENSE = True
            print("Selected: RealSense camera")
            break
        elif choice == 'B':
            USE_REALSENSE = False
            print("Selected: Notebook webcam")
            break
        else:
            if choice == 'A' and not REALSENSE_AVAILABLE:
                print("RealSense is not available on this system. Please select B.")
            else:
                print("Invalid choice. Please enter A or B.")
    
    # --- Initialization ---
    frame_count = 0
    data_to_save = []
    
    # Origin is automatically set at center of frame
    origin_set = True
    
    # Trajectory tracking variables
    previous_positions = []
    max_positions = 30
    draw_trajectories = False  # Only start drawing trajectories after pressing 'k'
    tracking_angle = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        target_class_id = -1
        if hasattr(model, 'names'):
            class_names = model.names
            for i, name in class_names.items():
                if name.lower() == TARGET_CLASS_NAME.lower():
                    target_class_id = i
                    print(f"Target class '{TARGET_CLASS_NAME}' found with ID: {target_class_id}")
                    break
            if target_class_id == -1:
                print(f"Warning: Target class '{TARGET_CLASS_NAME}' not found in model names: {class_names}. Model will detect all classes.")
        else:
            print("Warning: Could not access model class names. Model will detect all classes.")

    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Initialize camera - either RealSense or webcam
    if USE_REALSENSE:
        print("Initializing RealSense camera...")
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            
            config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            
            print("Starting RealSense pipeline...")
            profile = pipeline.start(config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth Scale is: {depth_scale}")
            
            # Setup alignment
            align_to = rs.stream.color
            align = rs.align(align_to)
            
            realsense_initialized = True
        except Exception as e:
            print(f"Error initializing RealSense: {e}")
            print("Falling back to regular webcam...")
            realsense_initialized = False
            USE_REALSENSE = False
    else:
        realsense_initialized = False
    
    # Initialize webcam if not using RealSense or if RealSense initialization failed
    if not USE_REALSENSE:
        print(f"Opening webcam (ID: {CAMERA_ID})...")
        cap = cv2.VideoCapture(CAMERA_ID)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam with ID {CAMERA_ID}")
            print("Trying to open default camera...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open any camera")
                exit()
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera properties: {actual_width}x{actual_height}, {actual_fps} FPS")

    # CSV File setup
    file_exists = os.path.exists(OUTPUT_CSV_FILE)
    csv_file = open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(CSV_HEADER)
        print(f"Created new CSV file: {OUTPUT_CSV_FILE}")
    else:
        print(f"Appending data to existing CSV file: {OUTPUT_CSV_FILE}")

    def estimate_z_from_bbox(bbox_width, bbox_height):
        """Estimate Z distance based on object size
        This is a rough estimation that needs calibration"""
        # Use the average of width and height estimates for more stability
        z_from_width = (REFERENCE_OBJECT_WIDTH * FOCAL_LENGTH) / bbox_width
        z_from_height = (REFERENCE_OBJECT_HEIGHT * FOCAL_LENGTH) / bbox_height
        z_est = (z_from_width + z_from_height) / 2.0
        return z_est

    def calculate_3d_distance(x, y, z):
        """Calculate 3D distance from origin"""
        return math.sqrt(x**2 + y**2 + z**2)

    def calculate_angle(point1, point2):
        """Calculate angle between two points relative to horizontal axis"""
        x1, y1 = point1
        x2, y2 = point2
        
        # Calculate angle in 2D space (X-Y plane)
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate angle in radians and convert to degrees
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg, dx, dy

    def draw_coordinate_system(image, orig_x, orig_y, size=50):
        """Draw coordinate system at origin point"""
        # X axis (red)
        cv2.arrowedLine(image, (orig_x, orig_y), (orig_x + size, orig_y), (0, 0, 255), 2)
        # Y axis (green)
        cv2.arrowedLine(image, (orig_x, orig_y), (orig_x, orig_y - size), (0, 255, 0), 2)
        # Z axis (blue, pointing out of the screen)
        cv2.circle(image, (orig_x, orig_y), 5, (255, 0, 0), -1)
        cv2.circle(image, (orig_x, orig_y), 10, (255, 0, 0), 1)
        
        cv2.putText(image, "X", (orig_x + size + 5, orig_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, "Y", (orig_x - 5, orig_y - size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "Z", (orig_x - 20, orig_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def draw_trajectory(image, positions, color=(255, 0, 255), thickness=2):
        """Draw the trajectory line based on previous positions"""
        if len(positions) < 2:
            return
            
        for i in range(1, len(positions)):
            cv2.line(image, positions[i-1], positions[i], color, thickness)

    def get_realsense_depth(depth_frame, x, y, width, height):
        """Get average depth from RealSense depth frame for a region"""
        if depth_frame is None:
            return 0.0
            
        # Get region of interest
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(FRAME_WIDTH, x + width), min(FRAME_HEIGHT, y + height)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
            
        # Get depth image for the region
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_roi = depth_image[y1:y2, x1:x2]
        
        # Filter out zero depths and calculate average
        valid_depths = depth_roi[depth_roi > 0]
        if valid_depths.size > 0:
            average_depth_mm = np.mean(valid_depths)
            average_depth_m = average_depth_mm / 1000.0
            return average_depth_m
        
        return 0.0

    # --- Main Loop ---
    print("\nStarting camera tracking with XYZ coordinates...")
    print("Press 'k' to enable trajectory tracking and angle calculation")
    if USE_REALSENSE:
        print("Using RealSense for depth data")
    else:
        print("Using webcam with estimated depth")
    
    # For tracking time and calculating FPS
    frame_times = []
    
    try:
        while True:
            start_time = time.time()
            
            # Get frame from camera (RealSense or webcam)
            if USE_REALSENSE:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print("Warning: Could not get depth or color frame.")
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                frame = color_image
            else:
                ret, frame = cap.read()
                depth_frame = None
                
                if not ret:
                    print("Error reading frame from webcam.")
                    break
                
            frame_count += 1
            
            # Create a copy for drawing
            display_image = frame.copy()
            
            # Get actual frame dimensions
            height, width = frame.shape[:2]
            
            # Origin is at center of frame
            origin_x = width // 2
            origin_y = height // 2

            # Draw the origin and coordinate system
            draw_coordinate_system(display_image, origin_x, origin_y)
            cv2.putText(display_image, "Origin (0,0,0)", (origin_x - 40, origin_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Run object detection
            results = model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
            
            current_frame_data = []

            # Process detection results
            for result in results: 
                boxes = result.boxes 

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    conf = float(box.conf[0])            
                    cls_id = int(box.cls[0])             

                    if target_class_id != -1 and cls_id != target_class_id:
                        continue 

                    # Calculate center of bounding box
                    center_box_x = (x1 + x2) // 2
                    center_box_y = (y1 + y2) // 2
                    
                    # Calculate coordinates relative to origin
                    rel_x = center_box_x - origin_x
                    rel_y = origin_y - center_box_y  # Inverted because Y goes down in image coordinates
                    
                    # Estimate Z based on bounding box size
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    z_est = estimate_z_from_bbox(bbox_width, bbox_height)
                    
                    # Get RealSense depth if available
                    if USE_REALSENSE and depth_frame:
                        realsense_depth = get_realsense_depth(depth_frame, x1, y1, bbox_width, bbox_height)
                    else:
                        realsense_depth = 0.0
                    
                    # Calculate 3D distance
                    distance_3d = calculate_3d_distance(rel_x, rel_y, z_est)
                    
                    # Track positions for trajectory if enabled
                    if draw_trajectories:
                        previous_positions.append((center_box_x, center_box_y))
                        if len(previous_positions) > max_positions:
                            previous_positions.pop(0)
                        
                        # Draw trajectory
                        draw_trajectory(display_image, previous_positions)
                        
                        # Calculate angle if tracking
                        angle_deg, dx, dy = calculate_angle((origin_x, origin_y), (center_box_x, center_box_y))
                        tracking_angle = angle_deg
                    
                    # Always draw line from origin to detected object
                    cv2.line(display_image, (origin_x, origin_y), (center_box_x, center_box_y), (0, 165, 255), 2)
                    
                    # Add to current frame data
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    current_frame_data.append([
                        timestamp, frame_count, x1, y1, x2, y2, f"{conf:.2f}", 
                        (x2 - x1) * (y2 - y1), rel_x, rel_y, f"{z_est:.2f}", 
                        f"{distance_3d:.2f}", f"{realsense_depth:.3f}"
                    ])

                    # Create base label
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    
                    # Add vector information
                    if USE_REALSENSE and realsense_depth > 0:
                        depth_text = f"RS:{realsense_depth:.3f}m"
                    else:
                        depth_text = f"Est:{z_est:.2f}m"
                        
                    vector_label = f"X:{rel_x:+d} Y:{rel_y:+d} Z:{depth_text} D:{distance_3d:.2f}m"
                    
                    # Draw bounding box and labels
                    color = (0, 255, 0)
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(display_image, vector_label, (x1, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Mark center of the detection
                    cv2.circle(display_image, (center_box_x, center_box_y), 4, (255, 0, 0), -1)
                    
                    # Draw 3D axes at the detected object to show orientation
                    axis_size = min(30, bbox_width // 3)
                    cv2.arrowedLine(display_image, (center_box_x, center_box_y), 
                                   (center_box_x + axis_size, center_box_y), (0, 0, 255), 1)  # X-axis
                    cv2.arrowedLine(display_image, (center_box_x, center_box_y), 
                                   (center_box_x, center_box_y - axis_size), (0, 255, 0), 1)  # Y-axis
                    
                    # Add vector information to side panel - only if trajectories are enabled
                    if draw_trajectories:
                        info_box_x = 10
                        info_box_y = 100
                        
                        # Draw semi-transparent background
                        overlay = display_image.copy()
                        cv2.rectangle(overlay, 
                                    (info_box_x, info_box_y), 
                                    (info_box_x + 250, info_box_y + 120), 
                                    (0, 0, 0), 
                                    -1)
                        
                        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
                        
                        # Add vector information to side panel
                        cv2.putText(display_image, f"Vector Information:", 
                                   (info_box_x + 10, info_box_y + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_image, f"X-coordinate: {rel_x:+d} px", 
                                   (info_box_x + 10, info_box_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_image, f"Y-coordinate: {rel_y:+d} px", 
                                   (info_box_x + 10, info_box_y + 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        if USE_REALSENSE and realsense_depth > 0:
                            cv2.putText(display_image, f"RealSense Z: {realsense_depth:.3f} m", 
                                       (info_box_x + 10, info_box_y + 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.putText(display_image, f"Z-estimate: {z_est:.2f} m", 
                                       (info_box_x + 10, info_box_y + 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                       
                        cv2.putText(display_image, f"3D Distance: {distance_3d:.2f} m", 
                                   (info_box_x + 10, info_box_y + 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display angle if tracking                        
                        cv2.putText(display_image, f"Angle: {tracking_angle:.1f}°", 
                                   (info_box_x + 120, info_box_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if current_frame_data:
                data_to_save.extend(current_frame_data)

            # Calculate and display FPS
            end_time = time.time()
            frame_time = end_time - start_time
            frame_times.append(frame_time)
            
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Display mode (RealSense or Webcam)
            mode_text = "RealSense" if USE_REALSENSE else "Notebook Camera"
            cv2.putText(display_image, f"Mode: {mode_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display trajectory status
            trajectory_status = "ON" if draw_trajectories else "OFF"
            cv2.putText(display_image, f"Trajectory Tracking: {trajectory_status}", 
                       (width - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if draw_trajectories else (0, 0, 255), 2)

            # Add instructions
            instruction_y = height - 10
            cv2.putText(display_image, "Press 'k': Toggle Trajectory | 's': Save Data | 'q': Quit", 
                       (10, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the frame
            window_title = f"XYZ Tracking - {mode_text}"
            cv2.imshow(window_title, display_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('k'):
                # Toggle trajectory drawing
                draw_trajectories = not draw_trajectories
                if draw_trajectories:
                    print("Trajectory tracking enabled")
                    # Clear previous positions when enabling
                    previous_positions = []
                else:
                    print("Trajectory tracking disabled")
                    previous_positions = []
                    
            elif key == ord('s'):
                if data_to_save:
                    print(f"\n--- Saving {len(data_to_save)} detection(s) to {OUTPUT_CSV_FILE} ---")
                    csv_writer.writerows(data_to_save)
                    csv_file.flush() 
                    print("--- Data saved successfully! ---")
                    data_to_save.clear() 
                else:
                    print("\n--- No new data to save. ---")

            elif key == ord('q'):
                print("\nExiting program...")
                break

    finally:
        # --- Cleanup ---
        if USE_REALSENSE:
            print("Stopping RealSense pipeline...")
            pipeline.stop()
        else:
            print("Releasing webcam...")
            cap.release()
            
        print("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        print("Closing CSV file...")
        if csv_file and not csv_file.closed:
            if data_to_save:
                print(f"\n--- Saving remaining {len(data_to_save)} detection(s) before closing... ---")
                csv_writer.writerows(data_to_save)
                csv_file.flush()
            csv_file.close()
        print("Cleanup finished.")

if __name__ == "__main__":
    main()