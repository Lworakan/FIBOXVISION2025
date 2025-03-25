import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import time
import os
from datetime import datetime

class DepthROIRecorder:
    def __init__(self):
        # ตั้งค่าตัวแปรสำหรับการตีกรอบ
        self.roi_point1 = None
        self.roi_point2 = None
        self.drawing = False
        self.roi_selected = False
        self.is_right_image = False  # เก็บว่า ROI อยู่ในภาพด้านขวา (depth) หรือไม่ (ใช้เพื่อการบันทึกข้อมูลเท่านั้น)
        
        # สร้างโฟลเดอร์สำหรับเก็บข้อมูล
        self.data_folder = "newEnvironmentTest\Realsense\collected_data\outdoor1stfloor"
        os.makedirs(self.data_folder, exist_ok=True)
        
        # ตัวแปรสำหรับการบันทึกข้อมูล
        self.recording = False 
        self.csv_filename = os.path.join(self.data_folder, f"depth_values_9m.csv")  # บันทึกข้อมูลทดสอบที่ระยะ 4m
        self.depth_values = []
        self.last_record_time = 0
        self.test_session = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # เก็บเวลาของการทดสอบปัจจุบัน
        self.current_avg_depth = None  # เก็บค่า depth เฉลี่ยปัจจุบัน
        
        # กำหนดขนาดภาพ
        self.width = 640
        self.height = 480
        
        # เริ่มต้นกล้อง RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # กำหนดการตั้งค่าสตรีม
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        # เริ่มต้นไปป์ไลน์
        self.profile = self.pipeline.start(self.config)
        
        # รับค่า depth scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"กำลังใช้งานกล้อง RealSense - Depth Scale: {self.depth_scale}")
        
        # สร้างหน้าต่างและตั้งค่าฟังก์ชันเมื่อคลิกเมาส์
        cv2.namedWindow('RealSense')
        cv2.setMouseCallback('RealSense', self.mouse_callback)
        
        # ตัวอย่างจากโค้ดที่มีอยู่ในโปรเจค:
        self.threshold_filter = rs.threshold_filter()
        self.max_distance = 16.0  # ค่าเริ่มต้น (เมตร)
        self.threshold_filter.set_option(rs.option.max_distance, self.max_distance)
    
    def normalize_coordinates(self, x, y):
        # แปลงพิกัดจากภาพรวมให้เป็นพิกัดในภาพเดี่ยว (ซ้ายหรือขวา)
        normalized_x = x
        normalized_y = y
        is_right = False
        
        # ตรวจสอบว่าคลิกที่ภาพซ้าย (สี) หรือภาพขวา (depth)
        if x >= self.width:
            normalized_x = x - self.width  # แปลงพิกัด x ให้อยู่ในช่วง 0-639
            is_right = True
        
        return normalized_x, normalized_y, is_right
    
    def mouse_callback(self, event, x, y, flags, param):
        # ฟังก์ชันรับการคลิกเมาส์สำหรับวาดกรอบ
        if event == cv2.EVENT_LBUTTONDOWN and not self.drawing:
            norm_x, norm_y, is_right = self.normalize_coordinates(x, y)
            self.roi_point1 = (norm_x, norm_y)
            self.roi_point2 = (norm_x, norm_y)
            self.drawing = True
            self.roi_selected = False
            self.is_right_image = is_right  # เก็บไว้เพื่อแสดงข้อมูลว่าคลิกที่ภาพไหน
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            norm_x, norm_y, _ = self.normalize_coordinates(x, y)
            self.roi_point2 = (norm_x, norm_y)
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            norm_x, norm_y, _ = self.normalize_coordinates(x, y)
            self.roi_point2 = (norm_x, norm_y)
            self.drawing = False
            self.roi_selected = True
            
            # แสดงข้อมูลพื้นที่ที่เลือก
            print(f"พื้นที่ ROI ที่เลือก: {self.roi_point1} ถึง {self.roi_point2}")
            print(f"ภาพที่เลือก: {'Depth (ขวา)' if self.is_right_image else 'Color (ซ้าย)'}")
            
    def get_normalized_roi(self):
        # รับค่าจุดกรอบที่ถูกต้อง (ซ้ายบน, ขวาล่าง)
        if not self.roi_selected or not self.roi_point1 or not self.roi_point2:
            return None
            
        x1, y1 = self.roi_point1
        x2, y2 = self.roi_point2
        
        # สลับค่าถ้าจำเป็นเพื่อให้ x1,y1 อยู่ที่มุมซ้ายบน และ x2,y2 อยู่ที่มุมขวาล่าง
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # ตรวจสอบว่าพิกัดอยู่ในขอบเขตของภาพหรือไม่
        x1 = max(0, min(x1, self.width-1))
        y1 = max(0, min(y1, self.height-1))
        x2 = max(0, min(x2, self.width-1))
        y2 = max(0, min(y2, self.height-1))
            
        return (x1, y1, x2, y2)
    
    def calculate_average_depth(self, depth_frame):
        # คำนวณค่า depth เฉลี่ยภายในกรอบที่เลือก
        roi = self.get_normalized_roi()
        if not roi:
            return None
            
        x1, y1, x2, y2 = roi
        
        # ใช้ numpy array แทนการวนลูปเพื่อเพิ่มประสิทธิภาพ
        depth_image = np.asanyarray(depth_frame.get_data())
        roi_depth = depth_image[y1:y2+1, x1:x2+1]
        
        # กรองค่า 0 ออกจากการคำนวณ
        valid_depth = roi_depth[roi_depth > 0]
        
        if len(valid_depth) == 0:
            return 0
        
        # คำนวณค่าเฉลี่ยและแปลงเป็นเมตร
        avg_depth = np.mean(valid_depth) * self.depth_scale
        return avg_depth
    
    def save_to_csv(self):
        # บันทึกข้อมูลลงไฟล์ CSV
        if len(self.depth_values) == 0:
            print("ไม่มีข้อมูลที่จะบันทึก")
            return
            
        # ตรวจสอบว่าไฟล์มีอยู่แล้วหรือไม่
        file_exists = os.path.isfile(self.csv_filename)
        
        # เปิดไฟล์ในโหมดเขียนต่อท้าย (append) ถ้าไฟล์มีอยู่แล้ว
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # เขียนหัวตารางเฉพาะเมื่อสร้างไฟล์ใหม่
            if not file_exists:
                writer.writerow(['session_time', 'timestamp', 'average_depth_m'])
            
            # เขียนข้อมูลพร้อมเวลาเริ่มต้นของการทดสอบครั้งนี้
            for value in self.depth_values:
                timestamp, depth = value
                writer.writerow([self.test_session, timestamp, depth])
        
        print(f"บันทึกข้อมูลแล้ว {len(self.depth_values)} ค่า ไปยังไฟล์ {self.csv_filename}")
    
    def run(self):
        try:
            print("เริ่มต้นโปรแกรม - คลิกและลากเพื่อเลือกพื้นที่สำหรับวัด depth")
            print("กด 's' เพื่อเริ่ม/หยุดการบันทึกค่า depth เฉลี่ย")
            print("กด 'q' เพื่อออกจากโปรแกรม")
            print("หมายเหตุ: คุณสามารถเลือกพื้นที่ในภาพสี (ซ้าย) หรือภาพ depth (ขวา) ก็ได้")
            print(f"ข้อมูลจะถูกบันทึกไปยัง: {self.csv_filename}")
            
            while True:
                # รอรับเฟรมจากกล้อง
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # แปลงเฟรมเป็น numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # สร้าง depth map แบบมีสี
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # สร้างสำเนาของภาพเพื่อวาดกรอบ
                color_display = color_image.copy()
                depth_display = depth_colormap.copy()
                
                # คำนวณ depth เฉลี่ยทุกครั้งหากมีการเลือก ROI
                if self.roi_selected:
                    self.current_avg_depth = self.calculate_average_depth(depth_frame)
                
                # แสดงกรอบ ROI ถ้ามีการวาด (แสดงทั้งสองภาพ)
                if self.drawing or self.roi_selected:
                    if self.roi_point1 and self.roi_point2:
                        roi = self.get_normalized_roi()
                        if roi:
                            x1, y1, x2, y2 = roi
                            
                            # สีของกรอบ: เขียวสำหรับกรอบที่ถูกเลือก, เหลืองสำหรับกรอบที่กำลังวาด
                            color = (0, 255, 0) if self.roi_selected else (0, 255, 255)
                            thickness = 2
                            
                            # วาดกรอบทั้งในภาพสีและภาพ depth
                            cv2.rectangle(color_display, (x1, y1), (x2, y2), color, thickness)
                            cv2.rectangle(depth_display, (x1, y1), (x2, y2), color, thickness)
                            
                            # แสดงข้อความ ROI บนกรอบทั้งสองภาพ
                            cv2.putText(color_display, "ROI", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(depth_display, "ROI", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # แสดงค่า depth เฉลี่ยทันทีหลังจากเลือก ROI เสร็จ
                            if self.roi_selected and self.current_avg_depth is not None:
                                depth_text = f"Depth: {self.current_avg_depth:.4f} m"
                                cv2.putText(color_display, depth_text, (10, 60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(depth_display, depth_text, (10, 60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # บันทึกค่า depth เฉลี่ยถ้ากำลังบันทึก
                if self.roi_selected and self.recording and self.current_avg_depth is not None:
                    # บันทึกค่าทุก 0.1 วินาที ไม่เกิน 2000 ค่า
                    current_time = time.time()
                    if current_time - self.last_record_time >= 0.1 and len(self.depth_values) < 2000:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        self.depth_values.append([timestamp, self.current_avg_depth])
                        self.last_record_time = current_time
                        print(f"บันทึกค่า: {self.current_avg_depth:.4f} m (ค่าที่ {len(self.depth_values)})")
                        
                        # หยุดบันทึกเมื่อได้ 1000 ค่า
                        if len(self.depth_values) >= 2000:
                            print("บันทึกครบ 1000 ค่าแล้ว หยุดการบันทึกอัตโนมัติ")
                            self.recording = False
                            self.save_to_csv()
                
                # รวมภาพสีและ depth map
                images = np.hstack((color_display, depth_display))
                
                # แสดงเส้นแบ่งระหว่างภาพซ้ายและขวา
                cv2.line(images, (self.width, 0), (self.width, self.height), (255, 255, 255), 1)
                
                # แสดงป้ายกำกับประเภทของภาพ
                cv2.putText(images, "Color", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(images, "Depth", (self.width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # แสดงข้อมูลการทดสอบที่ระยะ 4 เมตร
                test_info = "Depth Experiment"
                cv2.putText(images, test_info, (self.width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # แสดงสถานะการบันทึก
                status_text = f"บันทึก: {'เปิด' if self.recording else 'ปิด'}"
                if self.recording:
                    status_text += f" ({len(self.depth_values)}/1000)"
                cv2.putText(images, status_text, (10, images.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # แสดงภาพ
                cv2.imshow('RealSense', images)
                
                # รับการกดปุ่ม
                key = cv2.waitKey(1)
                
                # กด 'q' เพื่อออกจากโปรแกรม
                if key & 0xFF == ord('q'):
                    break
                
                # กด 's' เพื่อเริ่ม/หยุดการบันทึก
                elif key & 0xFF == ord('s'):
                    if self.roi_selected:
                        self.recording = not self.recording
                        if self.recording:
                            print("เริ่มบันทึกค่า depth")
                        else:
                            print("หยุดบันทึกค่า depth")
                            if len(self.depth_values) > 0:
                                self.save_to_csv()
                    else:
                        print("กรุณาเลือกพื้นที่ ROI ก่อนเริ่มบันทึกข้อมูล")
                
        finally:
            # ปิดการทำงานของกล้องและหน้าต่าง
            self.pipeline.stop()
            cv2.destroyAllWindows()
            # บันทึกข้อมูลก่อนปิดโปรแกรม (ถ้ามีข้อมูลที่ยังไม่ได้บันทึก)
            if self.recording and len(self.depth_values) > 0:
                self.save_to_csv()

if __name__ == "__main__":
    recorder = DepthROIRecorder()
    recorder.run() 