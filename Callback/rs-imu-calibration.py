import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.gyro)
config.enable_stream(rs.stream.accel)

profile = pipeline.start(config)

threshold_filter = rs.threshold_filter()
max_distance = 8.0 
threshold_filter.set_option(rs.option.max_distance, max_distance)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print("Depth Scale:", depth_scale)

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        gyro_frame = frames.first(rs.stream.gyro)
        accel_frame = frames.first(rs.stream.accel)

        if gyro_frame and accel_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            accel_data = accel_frame.as_motion_frame().get_motion_data()

            print(f"Gyroscope - X: {gyro_data.x}, Y: {gyro_data.y}, Z: {gyro_data.z}")
            print(f"Accelerometer - X: {accel_data.x}, Y: {accel_data.y}, Z: {accel_data.z}")

        if not depth_frame or not color_frame:
            continue

        filtered_depth_frame = threshold_filter.process(depth_frame)

        depth_image = np.asanyarray(filtered_depth_frame.get_data())

        depth_image_filtered = np.where(depth_image * depth_scale > max_distance, 0, depth_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_filtered, alpha=0.03), cv2.COLORMAP_JET)

        color_image = np.asanyarray(color_frame.get_data())

        stereo_image = np.hstack((color_image, depth_colormap))

        cv2.circle(stereo_image, (320, 240), 5, (0, 0, 255), -1)
        cv2.putText(stereo_image, f"Depth: {depth_image[240, 320] * depth_scale:.2f} m", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Stereo Image (Color + Depth)", stereo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
