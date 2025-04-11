import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_yolov11_model():
    try:
        model = YOLO('./Callback/best.pt')
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        return None

def calculate_depth_from_mask(mask, intrinsic_matrix):
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None, None, None
    
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    
    KNOWN_HOOP_DIAMETER = 0.45
    focal_length = intrinsic_matrix[0, 0]
    
    (_, radius) = cv2.minEnclosingCircle(largest_contour)
    diameter = 2 * radius
    
    depth = (focal_length * KNOWN_HOOP_DIAMETER) / diameter
    
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    x_3d = (centroid_x - cx) * depth / focal_length
    y_3d = (centroid_y - cy) * depth / focal_length
    z_3d = depth
    
    return (x_3d, y_3d, z_3d), (centroid_x, centroid_y), largest_contour, depth

def calculate_depth_from_bbox(bbox, intrinsic_matrix):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    KNOWN_HOOP_DIAMETER = 0.45
    focal_length = intrinsic_matrix[0, 0]
    
    depth = (focal_length * KNOWN_HOOP_DIAMETER) / max(w, h)
    
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    x_3d = (center_x - cx) * depth / focal_length
    y_3d = (center_y - cy) * depth / focal_length
    z_3d = depth
    
    return (x_3d, y_3d, z_3d), (center_x, center_y), depth

def process_video(video_path, output_path=None):
    model = load_yolov11_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {width}x{height} @ {fps}fps")
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    focal_length = 1000
    intrinsic_matrix = np.array([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ])
    
    plt.ion()
    fig = plt.figure(figsize=(10, 6))
    ax_3d = fig.add_subplot(111, projection='3d')
    positions = []
    
    frame_count = 0
    detection_count = 0
    segmentation_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}")
            
            display_frame = frame.copy()
            
            try:
                results = model.predict(frame, conf=0.1, verbose=False)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue
            
            for result in results:
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                
                for i, box in enumerate(result.boxes):
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    
                    if class_name == "basketball_hoop":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        has_segmentation = False
                        print(f"Processing detection {result}")
                        if hasattr(result, 'masks') and result.masks is not None:
                            try:
                                mask = result.masks[i]
                                
                                if hasattr(mask, 'data'):
                                    seg_mask = mask.data.cpu().numpy()
                                else:
                                    seg_mask = mask.cpu().numpy()
                                
                                if len(seg_mask.shape) > 2:
                                    seg_mask = seg_mask[0]
                                
                                if seg_mask.shape[:2] != (height, width):
                                    seg_mask = cv2.resize(seg_mask, (width, height), 
                                                     interpolation=cv2.INTER_NEAREST)
                                
                                bin_mask = (seg_mask > 0.5).astype(np.uint8)
                                
                                if np.any(bin_mask):
                                    has_segmentation = True
                                    segmentation_count += 1
                                    
                                    color_mask = np.zeros_like(display_frame)
                                    color_mask[bin_mask == 1] = [0, 255, 0]
                                    
                                    alpha = 0.4
                                    mask_area = (bin_mask == 1)
                                    display_frame[mask_area] = cv2.addWeighted(
                                        display_frame[mask_area], 1-alpha,
                                        color_mask[mask_area], alpha, 0
                                    )
                                    
                                    coords_3d, center, contour, depth = calculate_depth_from_mask(
                                        bin_mask, intrinsic_matrix
                                    )
                                    
                                    if contour is not None:
                                        cv2.drawContours(display_frame, [contour], 0, (255, 255, 0), 2)
                                        
                                    cv2.putText(display_frame, "Segmentation", (x1, y1 - 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            except Exception as e:
                                print(f"Error processing mask: {e}")
                                has_segmentation = False
                        
                        if not has_segmentation:
                            overlay = display_frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 0), -1)
                            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                            
                            coords_3d, center, depth = calculate_depth_from_bbox(
                                (x1, y1, x2, y2), intrinsic_matrix
                            )
                            
                            cv2.putText(display_frame, "Detection Only", (x1, y1 - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        if coords_3d is not None:
                            x_3d, y_3d, z_3d = coords_3d
                            center_x, center_y = center
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(display_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            position_text = f"X: {x_3d:.2f}m, Y: {y_3d:.2f}m, Z: {z_3d:.2f}m"
                            cv2.putText(display_frame, position_text, (x1, y2 + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            positions.append((x_3d, y_3d, z_3d))
                            detection_count += 1
                            
                            if frame_count % 10 == 0 and positions:
                                ax_3d.clear()
                                ax_3d.set_xlabel('X (m)')
                                ax_3d.set_ylabel('Y (m)')
                                ax_3d.set_zlabel('Z (m)')
                                ax_3d.set_title('Basketball Hoop 3D Position')
                                
                                recent_pos = positions[-20:] if len(positions) > 20 else positions
                                x_vals = [p[0] for p in recent_pos]
                                y_vals = [p[1] for p in recent_pos]
                                z_vals = [p[2] for p in recent_pos]
                                
                                ax_3d.scatter(x_vals, y_vals, z_vals, c='r', marker='o')
                                
                                if len(recent_pos) > 1:
                                    ax_3d.plot(x_vals, y_vals, z_vals, 'b-')
                                
                                plt.draw()
                                plt.pause(0.001)
            
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"Detections: {detection_count}", (10, height - 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, f"With segmentation: {segmentation_count}", (10, height - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Basketball Hoop Detection", display_frame)
            
            if output_path:
                out.write(display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        plt.close()
        
        print("\nProcessing Complete:")
        print(f"Processed {frame_count} frames")
        print(f"Total detections: {detection_count}")
        print(f"With segmentation: {segmentation_count}")
        print(f"Detection only: {detection_count - segmentation_count}")

def main():
    input_video = "./src/video/WIN_20250124_09_01_44_Pro.mp4"
    output_video = "output_basketball_detection.mp4"
    
    print("Basketball Hoop Detection and 3D Position Measurement")
    print("====================================================")
    print("Controls:")
    print("  - Press 'q' to quit")
    print()
    print("This script will:")
    print("1. Try to use instance segmentation if available")
    print("2. Fall back to detection only if segmentation fails")
    print("3. Calculate 3D position coordinates in either case")
    print()
    
    process_video(input_video, output_video)

if __name__ == "__main__":
    main()