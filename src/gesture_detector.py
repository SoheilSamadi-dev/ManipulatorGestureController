from typing import Optional, Tuple

import cv2
import mediapipe as mp

from .gesture_types import Gesture
from .gesture_utils import (
    finger_states,
    count_non_thumb_extended,
    is_palm_facing_camera,
    is_thumb_up,
    finger_splay_ratio,
    min_adjacent_extended_splay_ratio,
)


class GestureDetector:
    def __init__(self, max_num_hands: int = 1, detection_confidence: float = 0.6, tracking_confidence: float = 0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self.drawer = mp.solutions.drawing_utils
        self.drawer_style = mp.solutions.drawing_styles

    def close(self):
        self.hands.close()

    def detect(self, frame_bgr) -> Tuple[Optional[Gesture], any]:
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        gesture = None
        annotated = frame_bgr.copy()
        if results.multi_hand_landmarks:
            # Use first detected hand for simplicity
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = (
                results.multi_handedness[0].classification[0].label
                if results.multi_handedness else "Right"
            )

            # Draw landmarks
            self.drawer.draw_landmarks(
                annotated,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.drawer_style.get_default_hand_landmarks_style(),
                self.drawer_style.get_default_hand_connections_style(),
            )

            lm = hand_landmarks.landmark
            states = finger_states(lm, handedness_label)
            non_thumb = count_non_thumb_extended(states)
            thumb_up = is_thumb_up(lm, handedness_label)
            all_five = states["thumb"] and non_thumb == 4
            splay = finger_splay_ratio(lm) if all_five else 0.0
            min_adj_splay = min_adjacent_extended_splay_ratio(lm, states) if non_thumb >= 2 else 0.0

            # Priority: START (thumbs up) -> STOP (open palm facing) -> numbers
            if thumb_up and non_thumb == 0:
                gesture = Gesture.START
            elif all_five and is_palm_facing_camera(lm):
                # Distinguish STOP (fingers together) vs FIVE (fingers spread)
                if splay < 0.28:
                    gesture = Gesture.STOP
                else:
                    gesture = Gesture.FIVE
            else:
                if non_thumb in (1, 2, 3, 4) and not states["thumb"]:
                    # For 2+ ensure there is visible angle/gap between adjacent extended fingers
                    if non_thumb >= 2 and min_adj_splay < 0.20:
                        gesture = None
                    else:
                        gesture = Gesture.from_count(non_thumb)
                elif all_five:
                    # Not palm-facing; still treat as FIVE
                    gesture = Gesture.FIVE

            # Annotate with info
            info = f"Hand: {handedness_label}  Fingers: T{int(states['thumb'])}/I{int(states['index'])}/M{int(states['middle'])}/R{int(states['ring'])}/P{int(states['pinky'])}"
            cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            if all_five:
                cv2.putText(annotated, f"Splay: {splay:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2, cv2.LINE_AA)
            if non_thumb >= 2 and not states["thumb"]:
                cv2.putText(annotated, f"MinAdjSplay: {min_adj_splay:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 50), 2, cv2.LINE_AA)

        if gesture:
            cv2.putText(annotated, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        return gesture, annotated
