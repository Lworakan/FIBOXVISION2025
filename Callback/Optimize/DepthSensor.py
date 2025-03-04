"""
OOP for Depth Sensor with Callback Function
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

class DepthSensor:
    def __init__(self, save_file):
        """เริ่มต้นคลาสด้วยการตั้งค่ากล้องและไฟล์บันทึกข้อมูล"""
        # ตั้งค่ากล้อง
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # เปิดใช้งานสตรีม
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # เริ่มการทำงานของกล้อง
        self.profile = self.pipeline.start(self.config)
        
        # ตั้งค่าตัวกรอง
        self.threshold_filter = rs.threshold_filter()
        self.max_distance = 8.0  # ค่าเริ่มต้น (เมตร)
        self.threshold_filter.set_option(rs.option.max_distance, self.max_distance)
        
        # คำนวณ Depth Scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        # ตั้งค่าการบันทึกข้อมูล
        self.save_file = save_file
        self.click_data = []  # รายการเก็บข้อมูลการคลิก
        
        # ตั้งค่าหน้าต่าง OpenCV
        cv2.namedWindow('Depth (Filtered)')
        cv2.setMouseCallback('Depth (Filtered)', self._mouse_callback_wrapper)

    def _mouse_callback_wrapper(self, event, x, y, flags, param):
        """Wrapper สำหรับเรียกใช้เมธอด callback ของคลาส"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_callback(x, y)

    def _mouse_callback(self, x, y):
        """ประมวลผลเหตุการณ์คลิกเมาส์และบันทึกข้อมูล"""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        filtered_frame = self.threshold_filter.process(depth_frame)
        
        if 0 <= x < 640 and 0 <= y < 480:
            depth = filtered_frame.get_distance(x, y)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self.click_data.append((timestamp, x, y, depth))
            
            # บันทึกข้อมูลทันทีที่คลิก
            self._save_to_file(timestamp, x, y, depth)
            print(f"บันทึกข้อมูล: {timestamp}, ({x}, {y}), ความลึก: {depth:.3f} m")

    def _save_to_file(self, timestamp, x, y, depth):
        """บันทึกข้อมูลลงไฟล์แบบ append"""
        try:
            with open(self.save_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp}, {x}, {y}, {depth:.4f}\n")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

    def _process_frames(self):
        """ประมวลผลเฟรมและแสดงผลภาพ"""
        frames = self.pipeline.wait_for_frames()
        
        # ดึงและกรองเฟรม
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        
        # ประมวลผลเฟรมความลึก
        filtered_depth = self.threshold_filter.process(depth_frame)
        depth_image = np.asanyarray(filtered_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # สร้าง Depth Colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        # วาดข้อมูลการคลิกล่าสุด
        if self.click_data:
            last_click = self.click_data[-1]
            x, y = last_click[1], last_click[2]
            cv2.circle(depth_colormap, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(
                depth_colormap,
                f"{last_click[3]:.3f} m",
                (x + 15, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        return color_image, depth_colormap

    def run(self):
        """เมธอดหลักสำหรับรันโปรแกรม"""
        try:
            while True:
                color_img, depth_img = self._process_frames()
                if color_img is not None and depth_img is not None:
                    cv2.imshow('Color', color_img)
                    cv2.imshow('Depth (Filtered)', depth_img)
                
                # จบการทำงานเมื่อกด ESC หรือปิดหน้าต่าง
                key = cv2.waitKey(1)
                if key == 27 or cv2.getWindowProperty('Depth (Filtered)', cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("ปิดการทำงานและบันทึกข้อมูลเรียบร้อยแล้ว")

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    depth_sensor = DepthSensor(save_file='depth_measurements.txt')
    depth_sensor.run()