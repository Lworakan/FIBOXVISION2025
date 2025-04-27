import cv2
import os
import subprocess

def capture_video(filename='output_video.mp4', fps=20.0):
    # Set up video capture (default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

    print("Recording... Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)  # Write frame to video file

        cv2.imshow('Recording', frame)  # Show the recording window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Open Explorer and select the file
    file_path = os.path.abspath(filename)
    # subprocess.run(f'explorer /select,"{file_path}"')


capture_video()