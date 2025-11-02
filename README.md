Hand Gesture Recognition (OpenCV + MediaPipe)

Overview

- Detects and logs these gestures from your laptop camera:
  - START: Thumbs up (only thumb extended, pointing upward)
  - STOP: Open palm facing camera (all five fingers extended, palm toward camera)
  - Numbers 1–5 by finger count (1–4 are index–pinky without thumb; 5 is all five)
- Appends recognized gestures to a .txt file with timestamps.

Install

1) Python 3.9+ recommended.
2) Install dependencies:

   pip install -r requirements.txt

Run

   python -m src.main --camera 0 --output recognized_gestures.txt

Options

- --camera <index>       Camera index (default 0)
- --output <path>        Output .txt log file path (default recognized_gestures.txt)
- --no-display           Disable the preview window
- --stable-frames <N>    Frames a gesture must persist before logging (default 5)

Notes on Gesture Rules

- START (thumbs up): Only thumb extended and pointing up (relative to wrist), other fingers folded.
- STOP: All five fingers extended and palm estimated to face the camera (using a simple palm normal heuristic).
- Numbers:
  - 1–4: Number of extended non‑thumb fingers (thumb not extended) — avoids conflict with START.
  - 5: All five fingers extended but palm not confidently facing camera; when palm faces camera, STOP takes priority.

Tips

- Good lighting improves detection.
- Keep your hand roughly centered in the frame.
- If gestures log too frequently, increase --stable-frames (e.g., 7–10).

