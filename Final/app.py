from Callback import VisionPipeline
import cv2
import numpy as np
import time
import os
## ถ้าไม่มีอะไรผิดพลาดกล้องจะเปิดตลอดจาก init ตรงนี้
# pipeline = VisionPipeline(camera_type='webcam', enable_visualization=True)

# try:
#     while True:
#         tracking_data, _ = pipeline.process_single_frame()

        
#         if tracking_data['detected']:
#             x = tracking_data['rel_x']
#             y = tracking_data['rel_y']
#             z = tracking_data['z']
#             angle = tracking_data['angle']
#             confidence = tracking_data['confidence']

#             print(f"Tracking Data: x={x}, y={y}, z={z}, angle={angle}, confidence={confidence}")


            
# except KeyboardInterrupt:
#     print("Exiting...")
# finally:
#     pipeline.stop()


# for debugging
# realsense
pipeline = VisionPipeline(camera_type='webcam', enable_visualization=True, enable_save_video=True)

Out_depth = 0
angle_Out = 0

# Create a folder to save frames if it doesn't exist
output_folder = f"output_frames_{time.strftime('%Y%m%d_%H%M%S')}"  # Use current timestamp in folder name
os.makedirs(output_folder, exist_ok=True)
frames = []
while True:
    # tracking_data, vis_img = pipeline.process_single_frame_multiple()
    # Process tracking_data
    # ...

    all_objects, vis_img = pipeline.process_single_frame_multiple()

    
    # Show visualization
    if vis_img is not None:
        cv2.imshow("Debug View", vis_img)
        
        # Save the current frame to the folder
        frame_filename = os.path.join(output_folder, f"frame_{cv2.getTickCount()}.jpg")
        cv2.imwrite(frame_filename, vis_img)
        
        frames.append(vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.save_video(frames, "output7.mp4", fps=15)
            break
        object_count = {}
        for i, obj in enumerate(all_objects):
            # if obj['detected']:
                # print(obj)
                # #  label = f"{class_names[class_id]} {conf:.2f}"
                # # print(f"Detected Object {i+1}: {obj['class_name']}")
                # print(f"Detected Object {i+1}: {obj['class_id']}")
                # print(f"Object {i+1}:")
                # print(f"  X: {obj['rel_x']:.2f}")
                # print(f"  Y: {obj['rel_y']:.2f}")
                # print(f"  Z: {obj['z']:.2f}m")
                # print(f"  Angle: {obj['angle']:.2f}°")
                # print(f"  Confidence: {obj['conf']:.2f}")

            class_name = obj['class_name']
            if class_name in object_count:
                object_count[class_name] += 1
            else:
                object_count[class_name] = 1

            top_objects = sorted(object_count.items(), key=lambda x: x[1], reverse=True)[:10]

            if cv2.waitKey(1) &0xFF == ord('s'):
                for class_name, count in top_objects:
                    if obj['detected'] and obj['class_name'] == 'white ball':
                        Out_depth = obj['z']
                        angle_Out = obj['angle']
                        print(f"Out Depth: {Out_depth}, Angle: {angle_Out}")

                    else:
                        # get min depth
                        min_depth = min(obj['z'] for obj in all_objects if obj['detected'])
                        Out_depth = min_depth
                        angle_Out = obj['angle']
                        print(f"Out Depth: {Out_depth}, Angle: {angle_Out}")
            else:
                pass

  
pipeline.stop()
cv2.destroyAllWindows()

