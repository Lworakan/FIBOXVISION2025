"""
ใช้สำหรับทดสอบกล้อง RealSense
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# สร้าง pipeline
pipeline = rs.pipeline()
config = rs.config()

# เปิดใช้งานสตรีมความลึกและสี
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# เริ่มต้น pipeline
pipeline.start(config)

try:
    while True:
        # รอรับเฟรมจากกล้อง
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        .0

        # แปลงเฟรมเป็นอาเรย์ numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # แปลงภาพความลึกเป็นสี
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # แสดงภาพ
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_colormap)
        
        # ออกจากลูปเมื่อกดปุ่ม ESC
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    # ปิด pipeline
    pipeline.stop()
    cv2.destroyAllWindows()