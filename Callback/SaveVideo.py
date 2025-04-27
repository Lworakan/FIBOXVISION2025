import os
import cv2
import numpy as np

def save_video(frames, filename, fps=30, codec='XVID', fourcc=None):
    if fourcc is None:
        fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # Get the dimensions of the first frame
    height, width, _ = frames[0].shape
    
    # Create a VideoWriter object
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved as {filename}")

def capture_from_webcam(output_filename, fps=30):
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frames = []
    print("Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frames.append(frame)
        cv2.imshow('Webcam', frame)

        # Stop recording if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frames:
        save_video(frames, output_filename, fps=fps)
    else:
        print("No frames captured.")
