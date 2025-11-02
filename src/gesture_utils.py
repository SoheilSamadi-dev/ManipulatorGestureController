from typing import Dict, Tuple
import math


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
    # For non-thumb fingers, require a straighter posture to reduce false positives:
    # Enforce y-order (tip < dip < pip < mcp), since smaller y is higher on screen.
    m = 0.015
    if finger == "index":
        tip, dip, pip, mcp = landmarks[INDEX_TIP], landmarks[INDEX_DIP], landmarks[INDEX_PIP], landmarks[INDEX_MCP]
    elif finger == "middle":
        tip, dip, pip, mcp = landmarks[MIDDLE_TIP], landmarks[MIDDLE_DIP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_MCP]
    elif finger == "ring":
        tip, dip, pip, mcp = landmarks[RING_TIP], landmarks[RING_DIP], landmarks[RING_PIP], landmarks[RING_MCP]
    elif finger == "pinky":
        tip, dip, pip, mcp = landmarks[PINKY_TIP], landmarks[PINKY_DIP], landmarks[PINKY_PIP], landmarks[PINKY_MCP]
    else:
        raise ValueError("finger must be one of: index, middle, ring, pinky")

    return (
        (tip.y + m) < dip.y and
        (dip.y + m) < pip.y and
        (pip.y + m) < mcp.y
    )
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
    # Consider the thumb up if it is extended and points upward significantly relative to the wrist.
    wrist = landmarks[WRIST]
    thumb_tip = landmarks[THUMB_TIP]
    dx = thumb_tip.x - wrist.x
    dy = thumb_tip.y - wrist.y
    # Primary check: clearly above wrist
    if dy >= -0.06:
        return False
    # Secondary check: vertical dominance
    vert_ok = abs(dx) < abs(dy) * 0.8
    # Also accept cases where thumb is nearly vertical even if is_thumb_extended() is borderline
    return vert_ok or is_thumb_extended(landmarks, handedness_label)


def _dist2d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_width(landmarks) -> float:
    """Approximate palm width using distance between index and pinky MCP joints."""
    return _dist2d(landmarks[INDEX_MCP], landmarks[PINKY_MCP])


def finger_splay_ratio(landmarks) -> float:
    """Average adjacent fingertip separation normalized by palm width.
    Returns 0 if palm width is too small (shouldn't happen in practice).
    """
    pw = palm_width(landmarks)
    if pw <= 1e-6:
        return 0.0
    idx_tip, mid_tip, ring_tip, pinky_tip = (
        landmarks[INDEX_TIP], landmarks[MIDDLE_TIP], landmarks[RING_TIP], landmarks[PINKY_TIP]
    )
    d_im = _dist2d(idx_tip, mid_tip)
    d_mr = _dist2d(mid_tip, ring_tip)
    d_rp = _dist2d(ring_tip, pinky_tip)
    return (d_im + d_mr + d_rp) / 3.0 / pw


def min_adjacent_extended_splay_ratio(landmarks, states: Dict[str, bool]) -> float:
    """Minimum normalized separation among adjacent extended fingertips (I-M, M-R, R-P).
    Returns 0 if fewer than two adjacent extended fingers are present.
    """
    pw = palm_width(landmarks)
    if pw <= 1e-6:
        return 0.0
    tips = {
        "index": landmarks[INDEX_TIP],
        "middle": landmarks[MIDDLE_TIP],
        "ring": landmarks[RING_TIP],
        "pinky": landmarks[PINKY_TIP],
    }
    pairs = [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]
    dists = []
    for a, b in pairs:
        if states.get(a) and states.get(b):
            dists.append(_dist2d(tips[a], tips[b]) / pw)
    if not dists:
        return 0.0
    return min(dists)


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
