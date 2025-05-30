#!/usr/bin/env python3
# data_collector.py - Collect images and depth data from RealSense camera

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from camera_handler import CameraHandler
from object_detector import ObjectDetector
import config

class DataCollector:
    """
    Class to collect training data from RealSense camera
    Saves color images, depth data, and detection information when 's' is pressed
    """
    
    def __init__(self, output_dir="collected_data"):
        """
        Initialize the data collector
        
        Args:
            output_dir (str): Directory to save collected data
        """
        self.output_dir = output_dir
        self.setup_directories()
        
        # Initialize camera (RealSense only)
        self.camera = CameraHandler(source_type='realsense')
        if not self.camera.initialized:
            raise RuntimeError("Failed to initialize RealSense camera")
        
        # Initialize object detector for optional detection info
        self.detector = ObjectDetector()
        
        # Get camera dimensions
        self.width, self.height = self.camera.get_dimensions()
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2
        
        # Collection counters
        self.sample_count = 0
        self.session_start_time = datetime.now()
        
        print(f"Data Collector initialized:")
        print(f"- Output directory: {self.output_dir}")
        print(f"- Frame dimensions: {self.width}x{self.height}")
        print(f"- Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nControls:")
        print("- Press 's' to save current frame and depth data")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset sample counter")
        print("-" * 50)
    
    def setup_directories(self):
        """Create necessary directories for data storage"""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.color_dir = os.path.join(self.output_dir, "color_images_train")
        self.depth_dir = os.path.join(self.output_dir, "depth_data")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        print(f"Created data directories in: {self.output_dir}")
    
    def extract_depth_at_detection(self, depth_frame, detection_box, depth_scale):
        """
        Extract depth information from detection bounding box
        
        Args:
            depth_frame: RealSense depth frame
            detection_box: [x1, y1, x2, y2] bounding box coordinates
            depth_scale: RealSense depth scale factor
            
        Returns:
            dict: Depth statistics and raw depth data for the detection area
        """
        if depth_frame is None:
            return None
            
        x1, y1, x2, y2 = map(int, detection_box)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)
        
        try:
            # Convert depth frame to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Extract region of interest
            depth_roi = depth_image[y1:y2, x1:x2]
            
            # Get center point depth
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_depth_raw = depth_image[center_y, center_x]
            center_depth_meters = float(center_depth_raw * depth_scale)
            
            # Sample depth points around the detection area
            sample_points = []
            step_x = max(1, (x2 - x1) // 10)  # Sample 10 points across width
            step_y = max(1, (y2 - y1) // 10)  # Sample 10 points across height
            
            for y in range(y1, y2, step_y):
                for x in range(x1, x2, step_x):
                    if y < self.height and x < self.width:
                        raw_depth = depth_image[y, x]
                        depth_meters = float(raw_depth * depth_scale)
                        sample_points.append({
                            'pixel_x': int(x),
                            'pixel_y': int(y),
                            'depth_raw': int(raw_depth),
                            'depth_meters': depth_meters
                        })
            
            # Filter out zero/invalid depths
            valid_depths = depth_roi[depth_roi > 0]
            
            if valid_depths.size > 0:
                # Convert to meters
                depths_meters = valid_depths * depth_scale
                
                return {
                    'statistics': {
                        'mean_depth': float(np.mean(depths_meters)),
                        'median_depth': float(np.median(depths_meters)),
                        'min_depth': float(np.min(depths_meters)),
                        'max_depth': float(np.max(depths_meters)),
                        'std_depth': float(np.std(depths_meters)),
                        'valid_pixels': int(valid_depths.size),
                        'total_pixels': int(depth_roi.size),
                        'fill_ratio': float(valid_depths.size / depth_roi.size)
                    },
                    'realsense_depth_data': {
                        'center_point': {
                            'pixel_x': center_x,
                            'pixel_y': center_y,
                            'depth_raw': int(center_depth_raw),
                            'depth_meters': center_depth_meters
                        },
                        'sample_points': sample_points,
                        'depth_scale_used': depth_scale,
                        'bounding_box_depth': {
                            'roi_shape': [int(y2-y1), int(x2-x1)],
                            'mean_raw': float(np.mean(depth_roi[depth_roi > 0])) if valid_depths.size > 0 else 0.0,
                            'median_raw': float(np.median(depth_roi[depth_roi > 0])) if valid_depths.size > 0 else 0.0
                        }
                    }
                }
            else:
                return {
                    'statistics': {
                        'mean_depth': 0.0,
                        'median_depth': 0.0,
                        'min_depth': 0.0,
                        'max_depth': 0.0,
                        'std_depth': 0.0,
                        'valid_pixels': 0,
                        'total_pixels': int(depth_roi.size),
                        'fill_ratio': 0.0
                    },
                    'realsense_depth_data': {
                        'center_point': {
                            'pixel_x': center_x,
                            'pixel_y': center_y,
                            'depth_raw': int(center_depth_raw),
                            'depth_meters': center_depth_meters
                        },
                        'sample_points': sample_points,
                        'depth_scale_used': depth_scale,
                        'bounding_box_depth': {
                            'roi_shape': [int(y2-y1), int(x2-x1)],
                            'mean_raw': 0.0,
                            'median_raw': 0.0
                        }
                    }
                }
                
        except Exception as e:
            print(f"Error extracting depth data: {e}")
            return None
    
    def save_sample(self, color_frame, depth_frame, depth_scale):
        """
        Save color image, depth data, and metadata for current frame
        
        Args:
            color_frame: RGB image from RealSense
            depth_frame: Depth frame from RealSense
            depth_scale: Depth scale factor
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        sample_id = f"sample_{self.sample_count:04d}_{timestamp}"
        
        # Save color image
        color_filename = f"{sample_id}.jpg"
        color_path = os.path.join(self.color_dir, color_filename)
        cv2.imwrite(color_path, color_frame)
        
        # Save depth data as numpy array
        depth_filename = f"{sample_id}_depth.npy"
        depth_path = os.path.join(self.depth_dir, depth_filename)
        if depth_frame is not None:
            depth_image = np.asanyarray(depth_frame.get_data())
            np.save(depth_path, depth_image)
        
        # Run object detection
        detections = self.detector.detect(color_frame) if self.detector.initialized else []
        
        # Create metadata
        metadata = {
            'sample_id': sample_id,
            'timestamp': timestamp,
            'sample_count': self.sample_count,
            'session_start': self.session_start_time.isoformat(),
            'camera_info': {
                'width': self.width,
                'height': self.height,
                'depth_scale': depth_scale,
                'origin_x': self.origin_x,
                'origin_y': self.origin_y
            },
            'files': {
                'color_image': color_filename,
                'depth_data': depth_filename
            },
            'detections': []
        }
        
        # Process each detection
        for i, detection in enumerate(detections):
            center_x = (detection['box'][0] + detection['box'][2]) // 2
            center_y = (detection['box'][1] + detection['box'][3]) // 2
            
            # Calculate relative coordinates
            rel_x = float(center_x - self.origin_x)
            rel_y = float(center_y - self.origin_y)
            
            # Extract depth information for this detection
            depth_data = self.extract_depth_at_detection(depth_frame, detection['box'], depth_scale)
            
            detection_data = {
                'detection_id': i,
                'class_id': detection['class_id'],
                'confidence': detection['conf'],
                'bounding_box': {
                    'x1': detection['box'][0],
                    'y1': detection['box'][1],
                    'x2': detection['box'][2],
                    'y2': detection['box'][3],
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': detection['box'][2] - detection['box'][0],
                    'height': detection['box'][3] - detection['box'][1]
                },
                'relative_coordinates': {
                    'rel_x': rel_x,
                    'rel_y': rel_y
                },
                'depth_information': depth_data
            }
            
            metadata['detections'].append(detection_data)
        
        # Save metadata as JSON
        metadata_filename = f"{sample_id}_metadata.json"
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.sample_count += 1
        
        print(f"✓ Saved sample {self.sample_count}: {sample_id}")
        print(f"  - Color image: {color_filename}")
        print(f"  - Depth data: {depth_filename}")
        print(f"  - Detections: {len(detections)}")
        if detections and depth_data and 'statistics' in depth_data:
            avg_depth = depth_data['statistics']['mean_depth']
            center_depth = depth_data['realsense_depth_data']['center_point']['depth_meters']
            print(f"  - Avg depth: {avg_depth:.3f}m")
            print(f"  - Center depth: {center_depth:.3f}m")
        print("-" * 50)
    
    def create_visualization(self, color_frame, detections):
        """
        Create visualization with detection overlays
        
        Args:
            color_frame: RGB image
            detections: List of detection results
            
        Returns:
            numpy.ndarray: Visualization image
        """
        vis_img = color_frame.copy()
        
        # Draw origin point
        cv2.circle(vis_img, (self.origin_x, self.origin_y), 5, (0, 255, 255), -1)
        cv2.putText(vis_img, "Origin", (self.origin_x - 30, self.origin_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw detections
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection['box'])
            conf = detection['conf']
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(vis_img, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Draw line from origin to center
            cv2.line(vis_img, (self.origin_x, self.origin_y), (center_x, center_y), (255, 0, 0), 1)
            
            # Add detection info
            label = f"Det {i+1}: {conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add sample counter and instructions
        cv2.putText(vis_img, f"Samples: {self.sample_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, "Press 's' to save, 'q' to quit", (10, vis_img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def run_collection(self):
        """
        Main data collection loop
        """
        print("Starting data collection. Waiting for RealSense frames...")
        
        try:
            while True:
                # Get frame from RealSense
                ret, color_frame, depth_frame = self.camera.get_frame()
                if not ret or color_frame is None:
                    print("Failed to get frame from RealSense")
                    continue
                
                # Get depth scale
                depth_scale = self.camera.get_depth_scale()
                
                # Run detection for visualization
                detections = self.detector.detect(color_frame) if self.detector.initialized else []
                
                # Create visualization
                vis_img = self.create_visualization(color_frame, detections)
                
                # Display the frame
                cv2.imshow("Data Collection - RealSense", vis_img)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # Save current frame and data
                    self.save_sample(color_frame, depth_frame, depth_scale)
                    
                elif key == ord('r'):
                    # Reset sample counter
                    self.sample_count = 0
                    print("Sample counter reset to 0")
                    
                elif key == ord('q'):
                    # Quit
                    print("Exiting data collection...")
                    break
                    
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Create summary file
        summary = {
            'session_info': {
                'start_time': self.session_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_samples': self.sample_count,
                'output_directory': self.output_dir
            },
            'camera_info': {
                'type': 'realsense',
                'width': self.width,
                'height': self.height,
                'origin_x': self.origin_x,
                'origin_y': self.origin_y
            }
        }
        
        summary_path = os.path.join(self.output_dir, "session_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nData collection complete!")
        print(f"Total samples collected: {self.sample_count}")
        print(f"Data saved to: {self.output_dir}")
        print(f"Session summary: {summary_path}")

# Main execution
if __name__ == "__main__":
    try:
        # Create data collector
        collector = DataCollector(output_dir="training_data")
        
        # Start collection
        collector.run_collection()
        
    except Exception as e:
        print(f"Error: {e}")
        if 'collector' in locals():
            collector.cleanup()