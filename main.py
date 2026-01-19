import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import threading
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class FallDetector:
    """
    A fall detection system using pose estimation with YOLOv8.
    Optimized for Raspberry Pi 4 with Pi Camera.
    Detects falls by analyzing body pose and position changes.
    """
    
    def __init__(self, model_name="yolov8n-pose.pt", confidence_threshold=0.5, 
                 enable_gpio=False, buzzer_pin=17):
        """
        Initialize the fall detector.
        
        Args:
            model_name: YOLOv8 pose model to use
            confidence_threshold: Minimum confidence for detections
            enable_gpio: Enable GPIO for alerts
            buzzer_pin: GPIO pin for buzzer (default: 17)
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.prev_positions = {}
        self.fall_threshold = 0.5
        self.alert_triggered = False
        self.alert_cooldown = 2
        self.last_alert_time = 0
        self.consecutive_fall_frames = 0
        self.fall_frame_threshold = 2
        
        # GPIO setup for Raspberry Pi
        self.enable_gpio = enable_gpio and GPIO_AVAILABLE
        self.buzzer_pin = buzzer_pin
        if self.enable_gpio:
            self._setup_gpio()
        
        # Camera setup
        self.use_pi_camera = PICAMERA_AVAILABLE
        self.camera = None
        
    def _setup_gpio(self):
        """Setup GPIO pins for alerts."""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.buzzer_pin, GPIO.OUT)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            print(f"✓ GPIO buzzer initialized on pin {self.buzzer_pin}")
        except Exception as e:
            print(f"✗ GPIO setup failed: {e}")
            self.enable_gpio = False
    
    def _init_pi_camera(self):
        """Initialize Raspberry Pi Camera."""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (640, 480)}
            )
            self.camera.configure(config)
            self.camera.start()
            print("✓ Pi Camera initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Pi Camera initialization failed: {e}")
            return False
    
    def detect_fall(self, frame):
        """
        Detect falls in a frame using pose estimation.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (annotated_frame, is_fall_detected, person_count)
        """
        results = self.model(frame, conf=self.confidence_threshold)
        
        annotated_frame = frame.copy()
        is_fall_detected = False
        person_count = 0
        frame_has_fall = False
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                person_count = len(keypoints)
                
                for idx, (kpts, box) in enumerate(zip(keypoints, boxes)):
                    if len(kpts) >= 17:
                        nose = kpts[0]
                        shoulders = [kpts[5], kpts[6]]
                        hips = [kpts[11], kpts[12]]
                        ankles = [kpts[15], kpts[16]]
                        
                        if self._is_person_fallen(nose, shoulders, hips, ankles):
                            frame_has_fall = True
                            self.consecutive_fall_frames += 1
                            
                            cv2.rectangle(annotated_frame, 
                                        (int(box[0]), int(box[1])),
                                        (int(box[2]), int(box[3])),
                                        (0, 0, 255), 3)
                            cv2.putText(annotated_frame, "FALL DETECTED", 
                                      (int(box[0]), int(box[1]) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            self.consecutive_fall_frames = 0
                            
                            cv2.rectangle(annotated_frame,
                                        (int(box[0]), int(box[1])),
                                        (int(box[2]), int(box[3])),
                                        (0, 255, 0), 2)
                    
                    self._draw_keypoints(annotated_frame, kpts)
        else:
            self.consecutive_fall_frames = 0
        
        if self.consecutive_fall_frames >= self.fall_frame_threshold:
            is_fall_detected = True
        elif not frame_has_fall:
            self.consecutive_fall_frames = 0
        
        return annotated_frame, is_fall_detected, person_count
    
    def _is_person_fallen(self, nose, shoulders, hips, ankles):
        """
        Determine if a person has fallen based on body position.
        
        Improved logic: A person is considered fallen if multiple indicators are present:
        1. Head is below hip level (strong indicator)
        2. Head is very close to ground (low y-position)
        3. Body is horizontal (height < width significantly)
        4. Head-to-hips distance indicates prone position
        
        Args:
            nose: Nose keypoint [x, y, confidence]
            shoulders: List of shoulder keypoints
            hips: List of hip keypoints
            ankles: List of ankle keypoints
            
        Returns:
            bool: True if person appears to have fallen
        """
        # Filter out low confidence points (more lenient threshold)
        valid_shoulders = [s for s in shoulders if s[2] > 0.2]
        valid_hips = [h for h in hips if h[2] > 0.2]
        valid_ankles = [a for a in ankles if a[2] > 0.2]
        
        if not valid_hips or not valid_shoulders or nose[2] < 0.2:
            return False
        
        # Calculate average positions
        avg_hip_y = np.mean([h[1] for h in valid_hips])
        avg_hip_x = np.mean([h[0] for h in valid_hips])
        avg_shoulder_y = np.mean([s[1] for s in valid_shoulders])
        avg_shoulder_x = np.mean([s[0] for s in valid_shoulders])
        avg_ankle_y = np.mean([a[1] for a in valid_ankles]) if valid_ankles else avg_hip_y + 50
        
        nose_y = nose[1]
        nose_x = nose[0]
        
        # Calculate key metrics
        head_to_hip_distance = abs(nose_y - avg_hip_y)
        shoulder_to_hip_distance = abs(avg_shoulder_y - avg_hip_y)
        head_height_ratio = nose_y / avg_ankle_y if avg_ankle_y > 0 else 0
        
        # Horizontal spread (width of body)
        body_width = abs(avg_shoulder_x - nose_x)
        body_height = avg_ankle_y - nose_y if avg_ankle_y > nose_y else 50
        
        # Multiple fall indicators
        indicators = []
        
        # Indicator 1: Head is below hips (strongest sign of fall)
        if nose_y > avg_hip_y:
            indicators.append(1)
        
        # Indicator 2: Head is below hips by significant margin (person lying down)
        if nose_y > avg_hip_y - 30:
            indicators.append(1)
        
        # Indicator 3: Head is very close to ground (y > 80% of frame height)
        if head_height_ratio > 0.75:
            indicators.append(1)
        
        # Indicator 4: Body is more horizontal than vertical
        if body_width > 0 and body_height > 0:
            aspect_ratio = body_height / body_width
            if aspect_ratio < 1.0:  # More wide than tall = lying down
                indicators.append(1)
        
        # Indicator 5: Head-to-hip distance is small (not standing upright)
        if head_to_hip_distance < shoulder_to_hip_distance * 0.3:
            indicators.append(1)
        
        # Person is considered fallen if 2+ indicators are present
        is_fallen = len(indicators) >= 2
        
        return is_fallen
    
    def _draw_keypoints(self, frame, keypoints, radius=5):
        """Draw pose keypoints on frame."""
        for kpt in keypoints:
            if kpt[2] > 0.3:  # Confidence threshold
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)
    
    def process_video(self, video_source=None, output_path=None, display=True):
        """
        Process video stream for fall detection.
        Supports both webcam and Pi Camera.
        
        Args:
            video_source: Video file path or camera index (None for Pi Camera, 0 for webcam)
            output_path: Optional path to save output video
            display: Whether to display the video (disable for headless mode)
        """
        # Try to use Pi Camera if available
        if self.use_pi_camera and video_source is None:
            print("Attempting to use Raspberry Pi Camera...")
            if self._init_pi_camera():
                self._process_pi_camera(output_path, display)
                return
            else:
                print("Falling back to USB/default camera...")
                self.use_pi_camera = False
                video_source = 0  # Use default camera
        
        # Default to camera 0 if no source specified
        if video_source is None:
            video_source = 0
        
        # Use webcam as fallback
        self._process_webcam(video_source, output_path, display)
    
    def _process_pi_camera(self, output_path=None, display=True):
        """Process video from Pi Camera."""
        print("Starting Pi Camera processing...")
        
        frame_count = 0
        fall_count = 0
        
        try:
            while True:
                # Capture frame from Pi Camera
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                frame_count += 1
                
                # Process frame
                annotated_frame, is_fall, person_count = self.detect_fall(frame)
                
                # Update fall counter
                if is_fall:
                    fall_count += 1
                    current_time = time.time()
                    
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        print(f"⚠️  FALL DETECTED! (Frame: {frame_count})")
                        self.last_alert_time = current_time
                        self.trigger_alert()
                
                # Add info overlay
                cv2.putText(annotated_frame, f"People: {person_count}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Falls: {fall_count}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame if enabled
                if display:
                    cv2.imshow("Fall Detection - Pi Camera", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Throttle to ~15 FPS for Pi performance
                time.sleep(0.066)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            if self.camera:
                self.camera.close()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete!")
            print(f"Total frames: {frame_count}")
            print(f"Total falls detected: {fall_count}")
    
    def _process_webcam(self, video_source=0, output_path=None, display=True):
        """Process video from webcam or file."""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Fall Detection Started. Press 'q' to quit.")
        frame_count = 0
        fall_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                annotated_frame, is_fall, person_count = self.detect_fall(frame)
                
                # Update fall counter
                if is_fall:
                    fall_count += 1
                    current_time = time.time()
                    
                    # Trigger alert with cooldown
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        print(f"⚠️  FALL DETECTED! (Frame: {frame_count})")
                        self.last_alert_time = current_time
                        self.trigger_alert()
                
                # Add info overlay
                cv2.putText(annotated_frame, f"People: {person_count}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Falls: {fall_count}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                if display:
                    cv2.imshow("Fall Detection", annotated_frame)
                
                # Save frame if output writer is available
                if out:
                    out.write(annotated_frame)
                
                # Quit on 'q' key
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\nProcessing complete!")
            print(f"Total frames: {frame_count}")
            print(f"Total falls detected: {fall_count}")
    
    def trigger_alert(self):
        """
        Trigger alert system.
        Supports GPIO buzzer, console alerts, and can be extended for notifications.
        """
        # Print visual alert
        print("\n" + "="*50)
        print("🚨 FALL ALERT! Immediate assistance needed! 🚨")
        print("="*50 + "\n")
        
        # Trigger GPIO buzzer if available
        if self.enable_gpio:
            self._buzz_alert()
    
    def _buzz_alert(self):
        """Buzz the GPIO buzzer for alert."""
        try:
            # Buzz pattern: 3 short buzzes
            for _ in range(3):
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                time.sleep(0.2)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                time.sleep(0.1)
        except Exception as e:
            print(f"Buzzer error: {e}")
    
    def cleanup(self):
        """Cleanup GPIO and resources."""
        if self.enable_gpio:
            try:
                GPIO.cleanup()
                print("GPIO cleaned up")
            except Exception as e:
                print(f"GPIO cleanup error: {e}")
        
        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                print(f"Camera cleanup error: {e}")


def main():
    """Main entry point for the fall detection system."""
    import sys
    
    # Parse command line arguments
    enable_gpio = "--gpio" in sys.argv
    headless = "--headless" in sys.argv
    
    # Extract video source if provided
    video_source = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            video_source = arg if arg.isdigit() else (int(arg) if arg == "0" else arg)
            break
    
    print("="*50)
    print("Fall Detection System - Raspberry Pi Edition")
    print("="*50)
    
    if enable_gpio:
        print("✓ GPIO alerts enabled")
    if headless:
        print("✓ Headless mode (no display)")
    
    try:
        print("\nInitializing Fall Detection System...")
        detector = FallDetector(
            model_name="yolov8n-pose.pt",
            confidence_threshold=0.5,
            enable_gpio=enable_gpio,
            buzzer_pin=17
        )
        
        print("Starting video processing...")
        detector.process_video(
            video_source=video_source,
            display=not headless
        )
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    
    finally:
        if 'detector' in locals():
            detector.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()
