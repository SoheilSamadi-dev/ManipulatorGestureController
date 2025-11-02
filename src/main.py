import argparse
import sys
import time

try:
    import cv2
except Exception as e:
    print("Error: OpenCV (cv2) is required. Install with: pip install -r requirements.txt")
    raise

from .gesture_detector import GestureDetector
from .gesture_types import Gesture
from .event_logger import EventLogger


def run(camera_index: int, output_path: str, display: bool, min_stable_frames: int, fps_smooth: float = 0.9):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {camera_index}")
        sys.exit(1)

    detector = GestureDetector()
    logger = EventLogger(output_path)

    last_detected = None
    stable_count = 0
    last_logged = None
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera.")
                break

            start = time.time()
            gesture, annotated = detector.detect(frame)

            if gesture == last_detected and gesture is not None:
                stable_count += 1
            else:
                stable_count = 1 if gesture is not None else 0
                last_detected = gesture

            # Log only when gesture is stable for N frames and changed since last log
            if gesture is not None and stable_count >= min_stable_frames and gesture != last_logged:
                logger.log(gesture.value)
                last_logged = gesture

            # FPS overlay
            end = time.time()
            dt = end - start
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps = fps_smooth * fps + (1 - fps_smooth) * inst_fps
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            if display:
                cv2.imshow("Hand Gesture Recognition", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # No display, tiny sleep to avoid maxing CPU
                time.sleep(0.005)

    finally:
        detector.close()
        cap.release()
        if display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Hand gesture recognition: START (thumbs up), STOP (open palm), numbers 1-5")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--output", type=str, default="recognized_gestures.txt", help="Path to the output .txt log file")
    parser.add_argument("--no-display", action="store_true", help="Disable UI display window")
    parser.add_argument("--stable-frames", type=int, default=5, help="Frames a gesture must persist before logging")
    args = parser.parse_args()

    run(camera_index=args.camera, output_path=args.output, display=not args.no_display, min_stable_frames=args.stable_frames)


if __name__ == "__main__":
    main()

