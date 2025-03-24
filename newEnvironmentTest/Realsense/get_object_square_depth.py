"""
ไม่ใช้
"""

"""
โค้ดสำหรับเปิดกล้อง RealSense เพื่อเก็บข้อมูลความลึก (Depth)
โดยให้ผู้ใช้คลิกที่มุมทั้ง 4 ของสี่เหลี่ยม และคำนวณค่าเฉลี่ยความลึกภายในสี่เหลี่ยมนั้น

การใช้งาน:
1. ให้ผู้ใช้ป้อนชื่อสภาพแวดล้อมและระยะห่างจริง
2. คลิกที่มุมทั้ง 4 ของสี่เหลี่ยมในภาพ
3. กดปุ่ม 'S' เพื่อบันทึกข้อมูล หรือ 'R' เพื่อรีเซ็ตมุม
4. กดปุ่ม 'ESC' เพื่อออกจากโปรแกรม
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import statistics

class RectangleDepthCollector:
    def __init__(self, output_dir="./newEnvironmentTest/data"):
        """
        เริ่มต้นคลาสเก็บข้อมูลความลึกในรูปสี่เหลี่ยม
        
        Args:
            output_dir (str): โฟลเดอร์สำหรับบันทึกข้อมูล
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ตั้งค่าสำหรับเก็บข้อมูลพิกัดที่คลิก
        self.corners = []  # เก็บพิกัดมุมทั้ง 4
        self.actual_distance = None  # ระยะห่างจริง (ผู้ใช้ป้อน)
        self.environment_name = None  # ชื่อสภาพแวดล้อม
        
        # ตั้งค่ากล้อง RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # เริ่มต้นกล้อง
        self.profile = self.pipeline.start(self.config)
        
        # ตั้งค่าตัวกรอง
        self.threshold_filter = rs.threshold_filter()
        self.max_distance = 10.0  # เมตร
        self.threshold_filter.set_option(rs.option.max_distance, self.max_distance)
        
        # คำนวณ Depth Scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")
        
        # เตรียมหน้าต่างแสดงผล
        cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Color', self.mouse_callback)
        
        # สร้างไฟล์สำหรับเก็บข้อมูล
        self.setup_data_file()
    
    def setup_data_file(self):
        """
        ตั้งค่าไฟล์สำหรับเก็บข้อมูล
        """
        # ถามชื่อสภาพแวดล้อม
        self.environment_name = input("กรุณาระบุชื่อสภาพแวดล้อม (เช่น room1, outdoor): ")
        
        # สร้างไฟล์ CSV สำหรับเก็บข้อมูล
        self.data_file = os.path.join(self.output_dir, f"{self.environment_name}_rectangle_depth.csv")
        file_exists = os.path.exists(self.data_file)
        
        with open(self.data_file, 'a') as f:
            if not file_exists:
                f.write("timestamp,actual_distance,x1,y1,x2,y2,x3,y3,x4,y4,avg_depth,min_depth,max_depth,std_dev_depth\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        ฟังก์ชันรับการคลิกเมาส์เพื่อกำหนดมุมของสี่เหลี่ยม
        
        Args:
            event: ประเภทของเหตุการณ์เมาส์
            x, y: พิกัดที่คลิก
            flags, param: พารามิเตอร์เพิ่มเติม
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
                print(f"เพิ่มมุมที่ {len(self.corners)}: ({x}, {y})")
    
    def draw_rectangle(self, image):
        """
        วาดสี่เหลี่ยมจากมุมที่กำหนด
        
        Args:
            image: ภาพที่จะวาด
            
        Returns:
            image: ภาพที่วาดแล้ว
        """
        # คัดลอกภาพเพื่อไม่ให้กระทบต่อภาพต้นฉบับ
        img_with_rect = image.copy()
        
        # วาดจุดที่มุมทั้งหมด
        for i, corner in enumerate(self.corners):
            cv2.circle(img_with_rect, corner, 5, (0, 255, 0), -1)
            cv2.putText(img_with_rect, str(i+1), (corner[0]+10, corner[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # วาดเส้นเชื่อมระหว่างมุม
        if len(self.corners) > 1:
            for i in range(len(self.corners)-1):
                cv2.line(img_with_rect, self.corners[i], self.corners[i+1], (0, 255, 0), 2)
                
            # วาดเส้นปิดสี่เหลี่ยม (ถ้ามีมุมครบ 4 มุม)
            if len(self.corners) == 4:
                cv2.line(img_with_rect, self.corners[3], self.corners[0], (0, 255, 0), 2)
        
        return img_with_rect
    
    def calculate_average_depth(self, depth_frame):
        """
        คำนวณค่าเฉลี่ยความลึกภายในสี่เหลี่ยม
        
        Args:
            depth_frame: เฟรมความลึกจากกล้อง
            
        Returns:
            tuple: (ค่าเฉลี่ยความลึก, ค่าต่ำสุด, ค่าสูงสุด, ส่วนเบี่ยงเบนมาตรฐาน)
        """
        if len(self.corners) != 4:
            return None, None, None, None
        
        # หาค่า x, y ต่ำสุดและสูงสุดเพื่อกำหนดขอบเขตสี่เหลี่ยม
        x_coords = [corner[0] for corner in self.corners]
        y_coords = [corner[1] for corner in self.corners]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # สร้างหน้ากากสำหรับพื้นที่ในสี่เหลี่ยม
        mask = np.zeros((480, 640), dtype=np.uint8)
        points = np.array(self.corners, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # เก็บค่าความลึกของจุดที่อยู่ในสี่เหลี่ยม
        depths = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if mask[y, x] == 255:  # ตรวจสอบว่าจุดอยู่ในสี่เหลี่ยมหรือไม่
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:  # ตรวจสอบว่าค่าความลึกถูกต้อง
                        depths.append(depth)
        
        if depths:
            avg_depth = np.mean(depths)
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            std_dev = np.std(depths)
            return avg_depth, min_depth, max_depth, std_dev
        else:
            return None, None, None, None
    
    def save_data(self, avg_depth, min_depth, max_depth, std_dev):
        """
        บันทึกข้อมูลลงไฟล์ CSV
        
        Args:
            avg_depth: ค่าเฉลี่ยความลึก
            min_depth: ค่าความลึกต่ำสุด
            max_depth: ค่าความลึกสูงสุด
            std_dev: ส่วนเบี่ยงเบนมาตรฐานของความลึก
        """
        if avg_depth is None or self.actual_distance is None:
            print("ไม่สามารถบันทึกข้อมูลได้: ไม่มีข้อมูลความลึกหรือระยะห่างจริง")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # รวบรวมพิกัดของมุม
        corners_flat = []
        for corner in self.corners:
            corners_flat.extend([corner[0], corner[1]])
        
        # บันทึกข้อมูลลงไฟล์
        with open(self.data_file, 'a') as f:
            f.write(f"{timestamp},{self.actual_distance},")
            f.write(",".join(str(x) for x in corners_flat))
            f.write(f",{avg_depth:.4f},{min_depth:.4f},{max_depth:.4f},{std_dev:.4f}\n")
        
        print(f"บันทึกข้อมูลสำเร็จ: ระยะห่างจริง = {self.actual_distance}m, ค่าเฉลี่ยความลึก = {avg_depth:.4f}m")
    
    def run(self):
        """
        ฟังก์ชันหลักสำหรับเก็บข้อมูล
        """
        print("\n=== เริ่มการเก็บข้อมูลความลึกในรูปสี่เหลี่ยม ===")
        print("คำแนะนำ:")
        print("1. คลิกที่มุมทั้ง 4 ของสี่เหลี่ยมในภาพ Color")
        print("2. กดปุ่ม 'D' เพื่อป้อนระยะห่างจริง")
        print("3. กดปุ่ม 'S' เพื่อบันทึกข้อมูล")
        print("4. กดปุ่ม 'R' เพื่อรีเซ็ตมุม")
        print("5. กดปุ่ม 'ESC' เพื่อออกจากโปรแกรม")
        
        try:
            while True:
                # รับเฟรมจากกล้อง
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # แปลงเป็นอาเรย์ numpy
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # แสดงข้อมูลบนภาพ
                info_image = color_image.copy()
                
                # แสดงสถานะการเก็บข้อมูล
                cv2.putText(info_image, f"Environment: {self.environment_name}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(info_image, f"Actual Distance: {self.actual_distance if self.actual_distance else 'Not set'} m", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(info_image, f"Corners: {len(self.corners)}/4", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # คำนวณและแสดงค่าเฉลี่ยความลึก
                avg_depth, min_depth, max_depth, std_dev = None, None, None, None
                if len(self.corners) == 4:
                    avg_depth, min_depth, max_depth, std_dev = self.calculate_average_depth(depth_frame)
                    if avg_depth is not None:
                        cv2.putText(info_image, f"Avg Depth: {avg_depth:.4f} m", (20, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(info_image, f"Min: {min_depth:.4f} m, Max: {max_depth:.4f} m", (20, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(info_image, f"Std Dev: {std_dev:.4f} m", (20, 180),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # วาดสี่เหลี่ยม
                img_with_rect = self.draw_rectangle(info_image)
                
                # แปลงภาพความลึกเป็น colormap
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # แสดงภาพ
                cv2.imshow('Color', img_with_rect)
                cv2.imshow('Depth', depth_colormap)
                
                # รอการกดปุ่ม
                key = cv2.waitKey(1)
                
                # ตรวจสอบปุ่มที่กด
                if key == 27:  # ESC
                    break
                elif key == ord('r') or key == ord('R'):  # รีเซ็ตมุม
                    self.corners = []
                    print("รีเซ็ตมุมแล้ว")
                elif key == ord('s') or key == ord('S'):  # บันทึกข้อมูล
                    if len(self.corners) == 4 and self.actual_distance is not None:
                        self.save_data(avg_depth, min_depth, max_depth, std_dev)
                        self.corners = []  # รีเซ็ตหลังจากบันทึก
                    else:
                        if len(self.corners) != 4:
                            print("ต้องกำหนดมุมให้ครบ 4 มุมก่อนบันทึก")
                        if self.actual_distance is None:
                            print("ต้องกำหนดระยะห่างจริงก่อนบันทึก (กดปุ่ม 'D')")
                elif key == ord('d') or key == ord('D'):  # กำหนดระยะห่างจริง
                    distance_str = input("กรุณาระบุระยะห่างจริง (เมตร): ")
                    try:
                        self.actual_distance = float(distance_str)
                        print(f"กำหนดระยะห่างจริงเป็น {self.actual_distance} เมตร")
                    except ValueError:
                        print("ระยะห่างไม่ถูกต้อง กรุณาระบุเป็นตัวเลข")
                        
        finally:
            # ปิดการเชื่อมต่อกับกล้อง
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("ปิดการเชื่อมต่อกับกล้อง RealSense")

if __name__ == "__main__":
    collector = RectangleDepthCollector()
    collector.run()