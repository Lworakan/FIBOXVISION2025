import pyrealsense2 as rs
import csv
import time
import os
from datetime import datetime
import numpy as np
import cv2
import pandas as pd  # เพิ่ม import pandas
from pycaret.regression import load_model, predict_model # เพิ่ม import PyCaret

class DepthROIRecorder:
    def __init__(self, current_location='thirdfloor'): # เพิ่มพารามิเตอร์สำหรับระบุ Location
        # ตั้งค่าตัวแปรสำหรับการตีกรอบ
        self.roi_point1 = None
        self.roi_point2 = None
        self.drawing = False
        self.roi_selected = False
        # self.is_right_image = False # ไม่ได้ใช้โดยตรงในการทำนาย แต่เก็บไว้เผื่อ Debug

        # สร้างโฟลเดอร์สำหรับเก็บข้อมูล (ปรับ Path ตามต้องการ)
        # *** แก้ไข Path ตรงนี้ให้ถูกต้องบนเครื่องของคุณ ***
        # ตัวอย่าง Path แบบเต็ม (อาจจะต้องปรับ): 'C:/Users/YourUser/Documents/RealSenseData/outdoor1stfloor'
        self.data_folder = r"newEnvironmentTest\Realsense\calibration_data" # ใช้ raw string เพื่อจัดการ backslash
        os.makedirs(self.data_folder, exist_ok=True)

        # --- เพิ่มส่วนโหลดโมเดล PyCaret ---
        # *** แก้ไข Path ตรงนี้ให้ชี้ไปที่โมเดล .pkl ที่บันทึกไว้ ***
        self.model_path = r'newEnvironmentTest\Realsense\model\final_calibrated_depth_model.pkl'
        try:
            self.calibrated_model = load_model(self.model_path, verbose=False) # verbose=False ปิด output ตอนโหลด
            print(f"โหลดโมเดล Calibration สำเร็จจาก: {self.model_path}.pkl")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดโมเดล Calibration: {e}")
            print("*** คำเตือน: ไม่สามารถโหลดโมเดลได้ จะแสดงค่า Depth ที่วัดได้เท่านั้น ***")
            self.calibrated_model = None

        # --- กำหนด Location ปัจจุบัน ---
        # ตรวจสอบว่า Location ที่ระบุมา ตรงกับที่โมเดลเคยเทรนหรือไม่ (ควรจะเป็น)
        self.current_location = current_location
        print(f"Location ปัจจุบันตั้งค่าเป็น: {self.current_location}")

        # ตัวแปรสำหรับการบันทึกข้อมูล CSV (ยังคงบันทึกค่าดิบ)
        self.recording = False
        # *** แก้ไขชื่อไฟล์ CSV ตรงนี้ให้สอดคล้องกับระยะทางที่กำลังทดสอบ ***
        self.csv_filename = os.path.join(self.data_folder, f"depth_values_4m_test.csv") # เพิ่ม _test เพื่อไม่ให้ทับไฟล์เดิม
        self.depth_values = []
        self.last_record_time = 0
        self.test_session = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # เก็บเวลาของการทดสอบปัจจุบัน
        self.current_avg_depth = None  # เก็บค่า depth เฉลี่ยปัจจุบัน (ค่าดิบ)

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
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            print(f"เกิดข้อผิดพลาดในการเริ่มต้นกล้อง: {e}")
            print("โปรดตรวจสอบว่ากล้องเชื่อมต่ออยู่และไม่มีโปรแกรมอื่นใช้งานอยู่")
            exit() # ออกจากโปรแกรมถ้าเปิดกล้องไม่ได้

        # รับค่า depth scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"กำลังใช้งานกล้อง RealSense - Depth Scale: {self.depth_scale}")

        # สร้างหน้าต่างและตั้งค่าฟังก์ชันเมื่อคลิกเมาส์
        cv2.namedWindow('RealSense Calibration') # เปลี่ยนชื่อหน้าต่าง
        cv2.setMouseCallback('RealSense Calibration', self.mouse_callback)

        # ตัวกรองระยะทาง (Optional แต่มีประโยชน์)
        self.threshold_filter = rs.threshold_filter()
        self.max_distance = 16.0  # ตั้งค่าระยะทางสูงสุดที่สนใจ (เมตร)
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
            # self.is_right_image = is_right # ไม่ได้ใช้

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
            # print(f"ภาพที่เลือก: {'Depth (ขวา)' if self.is_right_image else 'Color (ซ้าย)'}")

    def get_normalized_roi(self):
        # รับค่าจุดกรอบที่ถูกต้อง (ซ้ายบน, ขวาล่าง)
        if not self.roi_selected or not self.roi_point1 or not self.roi_point2:
            return None

        x1, y1 = self.roi_point1
        x2, y2 = self.roi_point2

        # สลับค่าถ้าจำเป็น
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        # ตรวจสอบขอบเขต
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))

        # ตรวจสอบว่า ROI มีขนาดหรือไม่
        if x1 >= x2 or y1 >= y2:
            # print("Warning: ROI ไม่มีขนาด")
            return None

        return (x1, y1, x2, y2)

    def calculate_average_depth(self, depth_frame):
        # คำนวณค่า depth เฉลี่ยภายในกรอบที่เลือก
        roi = self.get_normalized_roi()
        if not roi:
            return None

        x1, y1, x2, y2 = roi

        try:
            # ใช้ numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            roi_depth = depth_image[y1:y2 + 1, x1:x2 + 1] # รวม y2, x2 ด้วย

            # กรองค่า 0 ออก
            valid_depth = roi_depth[roi_depth > 0]

            if valid_depth.size == 0: # ใช้ .size สำหรับ numpy array
                return 0.0 # คืนค่า 0.0 ถ้าไม่มีค่าที่วัดได้

            # คำนวณค่าเฉลี่ยและแปลงเป็นเมตร
            avg_depth_value = np.mean(valid_depth) * self.depth_scale
            return avg_depth_value
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการคำนวณ depth: {e}")
            return None

    def save_to_csv(self):
        # บันทึกข้อมูลลงไฟล์ CSV
        if not self.depth_values:
            print("ไม่มีข้อมูลที่จะบันทึก")
            return

        # ตรวจสอบว่าไฟล์มีอยู่แล้วหรือไม่
        file_exists = os.path.isfile(self.csv_filename)

        try:
            # เปิดไฟล์ในโหมดเขียนต่อท้าย (append)
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # เขียนหัวตารางเฉพาะเมื่อสร้างไฟล์ใหม่
                if not file_exists or os.path.getsize(self.csv_filename) == 0:
                    writer.writerow(['session_time', 'timestamp', 'average_depth_m'])

                # เขียนข้อมูล
                for value in self.depth_values:
                    timestamp, depth = value
                    writer.writerow([self.test_session, timestamp, depth])

            print(f"บันทึกข้อมูลแล้ว {len(self.depth_values)} ค่า ไปยังไฟล์ {self.csv_filename}")
            self.depth_values = [] # ล้าง buffer หลังจากบันทึก
        except IOError as e:
            print(f"เกิดข้อผิดพลาดในการเขียนไฟล์ CSV: {e}")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดที่ไม่คาดคิดในการบันทึก CSV: {e}")

    def run(self):
        try:
            print("เริ่มต้นโปรแกรม - คลิกและลากเพื่อเลือกพื้นที่สำหรับวัด depth")
            print("กด 's' เพื่อเริ่ม/หยุดการบันทึกค่า depth เฉลี่ย (ค่าดิบ)")
            print("กด 'q' เพื่อออกจากโปรแกรม")
            print(f"Location ปัจจุบัน: {self.current_location}")
            print(f"ข้อมูล CSV (ค่าดิบ) จะถูกบันทึกไปยัง: {self.csv_filename}")
            if not self.calibrated_model:
                print("*** ไม่ได้โหลดโมเดล Calibration ***")

            align_to = rs.stream.color
            align = rs.align(align_to)

            while True:
                # รอรับเฟรมและจัดตำแหน่ง
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # ใช้ตัวกรองระยะทาง (Optional)
                # depth_frame = self.threshold_filter.process(depth_frame)

                if not depth_frame or not color_frame:
                    continue

                # แปลงเฟรมเป็น numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # สร้าง depth map แบบมีสี
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), # ปรับ alpha ให้เห็นภาพชัดขึ้นได้
                    cv2.COLORMAP_JET
                )

                # สร้างสำเนาของภาพเพื่อวาดกรอบ
                color_display = color_image.copy()
                depth_display = depth_colormap.copy()

                # --- คำนวณและทำนาย ---
                calibrated_depth = None
                self.current_avg_depth = None # รีเซ็ตค่าก่อนคำนวณใหม่
                if self.roi_selected:
                    # คำนวณค่าเฉลี่ยดิบ
                    self.current_avg_depth = self.calculate_average_depth(depth_frame)

                    # ทำนายด้วยโมเดล ถ้าโหลดสำเร็จและมีค่า depth
                    if self.calibrated_model is not None and self.current_avg_depth is not None and self.current_avg_depth > 0:
                        try:
                            # สร้าง DataFrame สำหรับ Input
                            input_data = pd.DataFrame({
                                'average_depth_m': [self.current_avg_depth],
                                'Location': [self.current_location] # ใช้ Location ที่ตั้งค่าไว้
                            })
                            # ทำนาย
                            prediction = predict_model(self.calibrated_model, data=input_data, verbose=False)
                            # ดึงค่าที่ทำนายได้
                            calibrated_depth = prediction['prediction_label'].iloc[0]
                        except Exception as e:
                            print(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
                            calibrated_depth = None # ถ้าทำนายไม่ได้ ให้เป็น None

                # --- แสดงผล ---
                # วาดกรอบ ROI
                roi_coords = self.get_normalized_roi()
                if roi_coords:
                    x1, y1, x2, y2 = roi_coords
                    color = (0, 255, 0) if self.roi_selected else (0, 255, 255)
                    thickness = 2
                    cv2.rectangle(color_display, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(depth_display, (x1, y1), (x2, y2), color, thickness)
                    # cv2.putText(color_display, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    # cv2.putText(depth_display, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # แสดงค่า Depth (บนภาพสี)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color_measured = (0, 255, 255) # Yellow
                color_calibrated = (0, 255, 0)   # Green
                thickness_text = 2
                y_pos_measured = 60
                y_pos_calibrated = 90

                if self.current_avg_depth is not None:
                    measured_text = f"Measured : {self.current_avg_depth:.3f} m" # แสดง 3 ตำแหน่งทศนิยม
                    cv2.putText(color_display, measured_text, (10, y_pos_measured), font, font_scale, color_measured, thickness_text)
                    # cv2.putText(depth_display, measured_text, (10, y_pos_measured), font, font_scale, color_measured, thickness_text)

                if calibrated_depth is not None:
                    calibrated_text = f"Calibrated: {calibrated_depth:.3f} m" # แสดง 3 ตำแหน่งทศนิยม
                    cv2.putText(color_display, calibrated_text, (10, y_pos_calibrated), font, font_scale, color_calibrated, thickness_text)
                    # cv2.putText(depth_display, calibrated_text, (10, y_pos_calibrated), font, font_scale, color_calibrated, thickness_text)


                # --- บันทึกค่า depth เฉลี่ยดิบ ---
                if self.roi_selected and self.recording and self.current_avg_depth is not None:
                    current_time = time.time()
                    if current_time - self.last_record_time >= 0.1 and len(self.depth_values) < 2000:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        self.depth_values.append([timestamp, self.current_avg_depth])
                        self.last_record_time = current_time

                        if len(self.depth_values) >= 2000:
                            print("บันทึกครบ 2000 ค่าแล้ว หยุดการบันทึกอัตโนมัติ")
                            self.recording = False
                            self.save_to_csv()

                # --- รวมและแสดงภาพ ---
                images = np.hstack((color_display, depth_display))
                cv2.line(images, (self.width, 0), (self.width, self.height), (255, 255, 255), 1)
                cv2.putText(images, "Color", (10, 20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(images, "Depth", (self.width + 10, 20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(images, f"Location: {self.current_location}", (self.width - 250, 20), font, 0.5, (255, 255, 255), 1)

                # แสดงสถานะการบันทึก
                record_status_text = f"Record Raw: {'ON' if self.recording else 'OFF'}"
                record_color = (0, 0, 255) if self.recording else (0, 255, 0)
                if self.recording:
                    record_status_text += f" ({len(self.depth_values)}/2000)"
                cv2.putText(images, record_status_text, (10, images.shape[0] - 20), font, font_scale, record_color, thickness_text)

                cv2.imshow('RealSense Calibration', images)

                # --- รับ Input ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if self.roi_selected:
                        self.recording = not self.recording
                        if self.recording:
                            print("เริ่มบันทึกค่า depth ดิบ")
                        else:
                            print("หยุดบันทึกค่า depth ดิบ")
                            if self.depth_values: # บันทึกถ้ามีข้อมูล
                                self.save_to_csv()
                    else:
                        print("กรุณาเลือกพื้นที่ ROI ก่อนเริ่มบันทึกข้อมูล")

        except Exception as e:
             print(f"เกิดข้อผิดพลาดใน main loop: {e}")
        finally:
            # ปิดการทำงาน
            print("กำลังปิดกล้อง...")
            self.pipeline.stop()
            cv2.destroyAllWindows()
            # บันทึกข้อมูลที่เหลือ (ถ้ามี)
            if self.depth_values:
                print("บันทึกข้อมูล CSV ที่เหลือ...")
                self.save_to_csv()
            print("โปรแกรมสิ้นสุด")

if __name__ == "__main__":
    # --- กำหนด Location และ Path ที่นี่ ---
    # เลือก 'thirdfloor' หรือ 'outdoor1stfloor'
    location_to_run = 'thirdfloor'

    # *** สำคัญ: แก้ไข Path ไปยังโมเดล .pkl ของคุณ ***
    # model_directory = 'path/to/your/model/directory' # เช่น 'C:/Users/YourUser/Documents/PyCaretModels/'
    # model_name = 'final_calibrated_depth_model' # ชื่อไฟล์โมเดล (ไม่ต้องมี .pkl)

    # *** สำคัญ: แก้ไข Path สำหรับบันทึก CSV ***
    # csv_save_directory = f'path/to/your/data/directory/{location_to_run}' # เช่น 'C:/Users/YourUser/Documents/RealSenseData/thirdfloor'
    # csv_file_name_template = 'depth_values_{}m_test.csv' # {} จะถูกแทนที่ด้วยระยะ

    # --- สร้างและรัน Recorder ---
    # (ในโค้ดตัวอย่างนี้ Path ถูกกำหนดไว้ใน __init__, หากต้องการเปลี่ยนให้แก้ไขตรงนั้น หรือปรับแก้โค้ดส่วนนี้)
    recorder = DepthROIRecorder(current_location=location_to_run)
    recorder.run()