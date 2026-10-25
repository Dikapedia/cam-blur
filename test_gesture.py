import sys
import cv2
import numpy as np
import mediapipe as mp

from main import HandBlurCamera


def make_landmark(x, y, w, h):
    return type("L", (), {"x": x / w, "y": y / h, "z": 0.0})()


def build_landmarks(fingers="peace", hand_center=(320, 240), w=640, h=480):
    cx, cy = hand_center
    s = 30  # half spread per finger row

    # All fingers start at base level (wrist area), then extend/fold per case
    # Finger rows (tip -> PIP -> MCP -> wrist): higher y = lower on screen
    base = {
        # Wrist
        0: (cx, cy + s * 4),
        # Thumb
        1: (cx + s * 2, cy + s * 3),
        2: (cx + s * 3, cy + s * 2),
        3: (cx + s * 3, cy + s),
        4: (cx + s * 3, cy),
        # Index
        5: (cx - s * 2, cy + s * 3),
        6: (cx - s * 2, cy + s * 2),
        7: (cx - s * 2, cy + s),
        8: (cx - s * 2, cy),
        # Middle
        9: (cx - s, cy + s * 3),
        10: (cx - s, cy + s * 2),
        11: (cx - s, cy + s),
        12: (cx - s, cy),
        # Ring
        13: (cx + s, cy + s * 3),
        14: (cx + s, cy + s * 2),
        15: (cx + s, cy + s),
        16: (cx + s, cy),
        # Pinky
        17: (cx + s * 2, cy + s * 3),
        18: (cx + s * 2, cy + s * 2),
        19: (cx + s * 2, cy + s),
        20: (cx + s * 2, cy),
    }

    if fingers == "peace":
        base[8] = (cx - s * 2, cy - s * 2)   # index extended up
        base[12] = (cx - s, cy - s * 2)      # middle extended up
        base[16] = (cx + s, cy + s * 3)      # ring folded (below PIP)
        base[20] = (cx + s * 2, cy + s * 3)  # pinky folded (below PIP)
    elif fingers == "index_only":
        base[8] = (cx - s * 2, cy - s * 2)   # index extended
        base[12] = (cx - s, cy + s * 3)      # middle folded
        base[16] = (cx + s, cy + s * 3)      # ring folded
        base[20] = (cx + s * 2, cy + s * 3)  # pinky folded
    else:  # fist
        base[8] = (cx - s * 2, cy + s * 3)   # index folded
        base[12] = (cx - s, cy + s * 3)      # middle folded
        base[16] = (cx + s, cy + s * 3)      # ring folded
        base[20] = (cx + s * 2, cy + s * 3)  # pinky folded

    lms = []
    for i in range(21):
        px, py = base[i]
        lms.append(make_landmark(px, py, w, h))
    return lms


def test():
    cam = HandBlurCamera(resolution=(640, 480), blur_kernel=(15, 15))

    cases = [
        ("peace sign (2 fingers up)", "peace", True),
        ("index only (1 finger up)", "index_only", False),
        ("fist (0 fingers up)", "fist", False),
    ]

    fails = 0
    for name, fingers, expect in cases:
        lms = build_landmarks(fingers=fingers)
        got = cam._is_peace_sign(lms)
        status = "OK" if got == expect else "FAIL"
        if got != expect:
            fails += 1
        print(f"[{status}] {name}: expected={expect}, got={got}")

    print(f"\n{fails} failure(s)")
    return fails


if __name__ == "__main__":
    sys.exit(1 if test() else 0)
