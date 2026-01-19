#!/usr/bin/env python3
"""
Simple test to verify Pi Camera is working without YOLO
"""
import cv2
import time
from picamera2 import Picamera2

def test_camera():
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)
    
    print("Camera started. Capturing frames...")
    print("Press 'q' to quit")
    
    frame_count = 0
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_count += 1
            
            # Add counter and info
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Shape: {frame_bgr.shape}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Pi Camera Test", frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_count % 30 == 0:
                print(f"Captured {frame_count} frames")
                
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"Total frames: {frame_count}")

if __name__ == "__main__":
    test_camera()
