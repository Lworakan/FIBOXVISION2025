from callback import VisionPipeline

## ถ้าไม่มีอะไรผิดพลาดกล้องจะเปิดตลอดจาก init ตรงนี้
pipeline = VisionPipeline(camera_type='realsense', enable_visualization=False)

try:
    while True:
        tracking_data, _ = pipeline.process_single_frame()
        
        if tracking_data['detected']:
            x = tracking_data['rel_x']
            y = tracking_data['rel_y']
            z = tracking_data['z']
            angle = tracking_data['angle']
            confidence = tracking_data['confidence']
            
except KeyboardInterrupt:
    print("Exiting...")
finally:
    pipeline.stop()


# for debugging
# pipeline = VisionPipeline(camera_type='realsense', enable_visualization=True)

# while True:
#     tracking_data, vis_img = pipeline.process_single_frame()
    
#     # Process tracking_data
#     # ...
    
#     # Show visualization
#     if vis_img is not None:
#         cv2.imshow("Debug View", vis_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
# pipeline.stop()
# cv2.destroyAllWindows()