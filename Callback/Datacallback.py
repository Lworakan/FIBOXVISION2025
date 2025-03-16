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
        

        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.color = [640,480]
        self.config.enable_stream(rs.stream.depth,self.color[0], self.color[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.color[0], self.color[1], rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.gyro)
        self.config.enable_stream(rs.stream.accel)
        
        
        # เริ่มการทำงานของกล้อง
        self.profile = self.pipeline.start(self.config)
        
        # ตั้งค่าตัวกรอง
        self.threshold_filter = rs.threshold_filter()
        self.max_distance = 10.0  # ค่าเริ่มต้น (เมตร)
        self.threshold_filter.set_option(rs.option.max_distance, self.max_distance)
        self.spatial_filter = rs.spatial_filter()
        
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
            self._mouse_rectangle_callback(x, y, 20, 20)

        # self._mouse_callback(320, 240)

    def _mouse_rectangle_callback(self, x, y, w, h):
        """ประมวลผลเหตุการณ์คลิกเมาส์และบันทึกข้อมูล"""
        """หาค่าเฉลี่ยความลึด 1000 ค่าของสี่เหลี่ยม"""
        
        frame_count = 0  # Initialize frame count
        
        # Process up to 1000 frames
        Data_store = []
        while frame_count < 1000:
            # Wait for the next frame
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            filtered_frame = self.threshold_filter.process(depth_frame)
            depth_image = np.asanyarray(filtered_frame.get_data())
            gyro_frame = frames.first(rs.stream.gyro)
            accel_frame = frames.first(rs.stream.accel)

            # Apply spatial filter to depth frame
            filtered_depth_frame = self.spatial_filter.process(depth_frame)
            depth_image_filer = np.asanyarray(filtered_depth_frame.get_data())

            if 0 <= x < 640 and 0 <= y < 480:
                rect_depth = []
                # Loop through the rectangle to compute depth
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        # Calculate the depth for each pixel inside the rectangle
                        depth_value_filtered = depth_image_filer[j, i] * self.depth_scale
                        rect_depth.append(depth_value_filtered)

                # Compute the mean depth in the rectangle
                depth = np.mean(rect_depth)

                # Apply colormap for visualization
                avg_pixel_rect_depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                depth_image_filer_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image_filer, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Draw the rectangle on the image
                # cv2.rectangle(avg_pixel_rect_depth_colormap, (x, y), (x + w, y + h), (0, 255, 255), 2)
                x = 320
                y = 240
                w = 20
                h = 20
                cv2.rectangle(depth_image_filer_colormap, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Add text to the rectangle showing the depth
                cv2.putText(
                    depth_image_filer_colormap,
                    f"{depth:.3f} m",
                    (x + 15, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                cv2.circle(depth_image_filer_colormap, (x + w // 2, y + h // 2), 2, (0, 255, 255), -1)
                cv2.imshow('Depth spatial_filter (Filtered)_ save', depth_image_filer_colormap)

                # Get gyro and accelerometer data
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                depth_image_filer_list = depth_image_filer.tolist()
                # Data_store.append((timestamp, x, y, w, h, depth, depth_image_filer_list, gyro_data, accel_data))


            self._save_to_file(timestamp, x, y, w, h, depth, gyro_data, accel_data)


            frame_count += 1 

            key = cv2.waitKey(1)
            if key == 27 or cv2.getWindowProperty('Depth (Filtered)', cv2.WND_PROP_VISIBLE) < 1:  # ESC key to exit
                break
        # self._save_to_file(Data_store)
        cv2.destroyAllWindows()

        # print(f"บันทึกข้อมูล: {timestamp}, ({x}, {y}), ความลึก: {depth:.3f} m, depth_image_filer: {depth_image_filer_list}, Gyro: ({gyro_data.x}, {gyro_data.y}, {gyro_data.z}), Accel: ({accel_data.x}, {accel_data.y}, {accel_data.z})")
        # cv2.destroyAllWindows()
        


    def _mouse_callback(self, x, y):
        """ประมวลผลเหตุการณ์คลิกเมาส์และบันทึกข้อมูล"""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        filtered_frame = self.threshold_filter.process(depth_frame)
        depth_image = np.asanyarray(filtered_frame.get_data())

        gyro_frame = frames.first(rs.stream.gyro)
        accel_frame = frames.first(rs.stream.accel)

        
        if 0 <= x < x and 0 <= y < y:
            depth_frame = frames.get_depth_frame()

            # depth_frame = depth_frame[0]
            depth = (depth_frame.get_distance(x, y))
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            # self.click_data.append((timestamp, 320, 240, depth))

            
            # บันทึกข้อมูลทันทีที่คลิก
            # self._save_to_file(timestamp, x, y, depth, gyro_data, accel_data)
            # print(f"บันทึกข้อมูล: {timestamp}, ({x}, {y}), ความลึก: {depth:.3f} m, Gyro: ({gyro_data.x}, {gyro_data.y}, {gyro_data.z}), Accel: ({accel_data.x}, {accel_data.y}, {accel_data.z})")

    # self._save_to_file(timestamp, x, y, w, h, depth,depth_image_filer_list, gyro_data, accel_data)
    # timestamp, x, y, w, h, depth,depth_image_filer_list, gyro_data, accel_data):
    def _save_to_file(self, timestamp, x, y, w, h, depth, gyro_data, accel_data):
        """บันทึกข้อมูลลงไฟล์"""
        with open(self.save_file, 'a') as file:
            file.write(f"{timestamp},{x},{y},{w},{h},{depth},{gyro_data.x},{gyro_data.y},{gyro_data.z},{accel_data.x},{accel_data.y},{accel_data.z}\n")
        print(f"บันทึกข้อมูลลงไฟล์ {self.save_file} แล้ว")

    def _process_frames(self):
        """ประมวลผลเฟรมและแสดงผลภาพ"""
        frames = self.pipeline.poll_for_frames()

        # ดึงและกรองเฟรม
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None  # Make sure to return three values in case of an error

        # Apply the spatial filter to the depth frame
        filtered_depth_frame = self.spatial_filter.process(depth_frame)
        depth_image_filer = np.asanyarray(filtered_depth_frame.get_data())
        color_image_filer = np.asanyarray(color_frame.get_data())

        # Process the filtered depth frame
        filtered_depth = self.threshold_filter.process(depth_frame)
        depth_image = np.asanyarray(filtered_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create Depth Colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        depth_colormap_filer = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image_filer, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Drawing the click data rectangle
        x = self.color[0] // 2
        y = self.color[1] // 2
        w = 20
        h = 20
        depth = frames.get_depth_frame()

        depth_image_filer = np.asanyarray(filtered_depth_frame.get_data())

        depth_value_filtered = depth_image_filer[y, x] * self.depth_scale 

        print(f"Filtered Depth at ({x}, {y}): {depth_value_filtered} meters")

        cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(depth_colormap_filer, (x, y), (x + w, y + h), (255, 255, 255), 2)
        middle_x = x + w // 2
        middle_y = y + h // 2
        cv2.circle(depth_colormap, (middle_x, middle_y), 2, (255, 255, 255), -1)
        cv2.putText(
            depth_colormap,
            f"{depth_value_filtered:.3f} m",
            (x + 15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.circle(depth_colormap_filer, (middle_x, middle_y), 2, (255, 255, 255), -1)
        cv2.putText(
            depth_colormap_filer,
            f"{depth_value_filtered:.3f} m",
            (x + 15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Return three values
        return color_image, depth_colormap, depth_colormap_filer


    def run(self):
        """เมธอดหลักสำหรับรันโปรแกรม"""
        try:
            while True:
                color_img,depth_img, color_filer = self._process_frames()
                if color_img is not None and depth_img is not None:
                    cv2.imshow('Color', color_img)
                    cv2.imshow('Depth (Filtered)', depth_img)
                    cv2.imshow('Depth spatial_filter (Filtered)', color_filer)
                
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