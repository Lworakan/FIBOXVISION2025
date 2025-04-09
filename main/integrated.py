import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
import datetime
import csv
import os
import pandas as pd

# --- Configuration ---
# MODEL_PATH = r"Test_Realtime\best(1).pt" # model path
MODEL_PATH = r"./Callback/yolo11l.pt" # model path
OUTPUT_CSV_FILE = 'realsense_detections3500.csv' # output csv file
# -----------------------------

# YOLO Model Configuration
CONF_THRESHOLD = 0.60  # Confidence threshold
TARGET_CLASS_NAME = 'basketball_hoop' 

# RealSense Camera Configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# CSV Header
CSV_HEADER = ['Timestamp', 'Frame', 'X_min', 'Y_min', 'X_max', 'Y_max', 'Confidence', 'Average_Depth_m', 'area']

# --- Initialization ---
frame_count = 0
data_to_save = [] 

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

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

print("Starting RealSense pipeline...")
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

file_exists = os.path.exists(OUTPUT_CSV_FILE)
csv_file = open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow(CSV_HEADER)
    print(f"Created new CSV file: {OUTPUT_CSV_FILE}")
else:
    print(f"Appending data to existing CSV file: {OUTPUT_CSV_FILE}")


# --- Main Loop ---
print("\nStarting real-time detection... Press 's' to save data, 'q' to quit.")
try:
    while True:
        frame_count += 1
        start_time = time.time()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("Warning: Could not get depth or color frame.")
            continue

        depth_image_raw = np.asanyarray(depth_frame.get_data()) 
        color_image = np.asanyarray(color_frame.get_data()) 

        results = model(color_image, conf=CONF_THRESHOLD, device=device, verbose=False)

        current_frame_data = []

        for result in results: 
            boxes = result.boxes 

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                conf = float(box.conf[0])            
                cls_id = int(box.cls[0])             

                if target_class_id != -1 and cls_id != target_class_id:
                    continue 

                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(FRAME_WIDTH, x2), min(FRAME_HEIGHT, y2)

                if x1_c < x2_c and y1_c < y2_c: 
                    depth_roi = depth_image_raw[y1_c:y2_c, x1_c:x2_c]

                    valid_depths = depth_roi[depth_roi > 0]

                    if valid_depths.size > 0:
                        average_depth_mm = np.mean(valid_depths)
                        average_depth_m = average_depth_mm / 1000.0
                        depth_text = f"{average_depth_m:.2f}m"

                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        current_frame_data.append([
                            timestamp, frame_count, x1, y1, x2, y2, f"{conf:.2f}", f"{average_depth_m:.3f}", (x2 - x1) * (y2 - y1)
                        ])

                    else:
                        average_depth_m = 0.0
                        depth_text = "N/A"
                else:
                    average_depth_m = 0.0
                    depth_text = "ROI Invalid"

                label = f"{model.names[cls_id]} {conf:.2f} | Depth: {depth_text}"
                color = (0, 255, 0)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if current_frame_data:
            data_to_save.extend(current_frame_data)

        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("RealSense YOLOv11 Detection", color_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
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
    print("Stopping RealSense pipeline...")
    pipeline.stop()
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