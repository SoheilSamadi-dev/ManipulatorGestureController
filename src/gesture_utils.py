from typing import Dict, Tuple


# MediaPipe Hands landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def _v(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def is_palm_facing_camera(landmarks) -> bool:
    # Approximate palm normal from wrist->index_mcp and wrist->pinky_mcp
    wrist = landmarks[WRIST]
    vec_i = _v(wrist, landmarks[INDEX_MCP])
    vec_p = _v(wrist, landmarks[PINKY_MCP])
    nx, ny, nz = _cross(vec_i, vec_p)
    # In MediaPipe, z increases away from camera. Negative normal z roughly points toward camera.
    return nz < 0


def is_finger_extended(landmarks, finger: str) -> bool:
    # For non-thumb fingers, consider tip above PIP in image coordinates (y smaller means up)
    margin = 0.02
    if finger == "index":
        return (landmarks[INDEX_TIP].y + margin) < landmarks[INDEX_PIP].y
    if finger == "middle":
        return (landmarks[MIDDLE_TIP].y + margin) < landmarks[MIDDLE_PIP].y
    if finger == "ring":
        return (landmarks[RING_TIP].y + margin) < landmarks[RING_PIP].y
    if finger == "pinky":
        return (landmarks[PINKY_TIP].y + margin) < landmarks[PINKY_PIP].y
    raise ValueError("finger must be one of: index, middle, ring, pinky")


def is_thumb_extended(landmarks, handedness_label: str) -> bool:
    # For thumb, use x relation depending on handedness
    # Right hand: tip.x < ip.x when extended; Left hand: tip.x > ip.x when extended (camera coords)
    margin = 0.02
    tip_x = landmarks[THUMB_TIP].x
    ip_x = landmarks[THUMB_IP].x
    if handedness_label.lower().startswith("right"):
        return tip_x < (ip_x - margin)
    else:
        return tip_x > (ip_x + margin)


def is_thumb_up(landmarks, handedness_label: str) -> bool:
    if not is_thumb_extended(landmarks, handedness_label):
        return False
    wrist = landmarks[WRIST]
    thumb_tip = landmarks[THUMB_TIP]
    dx = thumb_tip.x - wrist.x
    dy = thumb_tip.y - wrist.y
    # Up means dy negative and dominant over dx
    if dy >= -0.05:
        return False
    return abs(dx) < abs(dy) * 0.5


def finger_states(landmarks, handedness_label: str) -> Dict[str, bool]:
    return {
        "thumb": is_thumb_extended(landmarks, handedness_label),
        "index": is_finger_extended(landmarks, "index"),
        "middle": is_finger_extended(landmarks, "middle"),
        "ring": is_finger_extended(landmarks, "ring"),
        "pinky": is_finger_extended(landmarks, "pinky"),
    }


def count_non_thumb_extended(states: Dict[str, bool]) -> int:
    return int(states["index"]) + int(states["middle"]) + int(states["ring"]) + int(states["pinky"])

