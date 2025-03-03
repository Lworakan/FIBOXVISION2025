import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

threshold_filter = rs.threshold_filter()
max_distance = 8.0
threshold_filter.set_option(rs.option.max_distance, max_distance)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print("Depth Scale:", depth_scale)

clicked_point = None
depth_value = 0

def mouse_callback(event, x, y, flags, param):
    global clicked_point, depth_value
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame = param[0]
        depth_value = (depth_frame.get_distance(x, y)) # ค่าความลึกที่ตำแหน่ง (x, y)
        clicked_point = (x, y)
        # print(f"Depth at ({x}, {y}): {depth_value:.2f} meters")

try:

    while True:
            # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame : continue

            filtered_depth_frame = threshold_filter.process(depth_frame)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_image_filtered = np.where(depth_image * depth_scale > max_distance, 0, depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_filtered, alpha=0.03), cv2.COLORMAP_JET)

            stereo_image = np.hstack((color_image, depth_colormap))


            if clicked_point:
                 cv2.putText(depth_colormap, f"{depth_value:.2f} m", (clicked_point[0], clicked_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Depth Image', depth_colormap)
            cv2.setMouseCallback('Depth Image', mouse_callback, param=[depth_frame])
            # cv2.imshow("Stereo Image (Color + Depth)", stereo_image)

            if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()