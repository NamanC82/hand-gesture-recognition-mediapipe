import numpy as np


def is_thumbs_up(landmark_list) -> bool:
    """
    Detects if the hand landmarks form a clear 'Thumbs Up' pose:
    - Thumb is pointing upwards and extended.
    - Other 4 fingers (Index, Middle, Ring, Pinky) are folded in.
    """
    if not landmark_list or len(landmark_list) < 21:
        return False

    pts = np.array(landmark_list, dtype=np.float64)
    wrist = pts[0]

    # 1. Check folded state of Index (8), Middle (12), Ring (16), Pinky (20)
    # Tips should be closer to the wrist or lower than their PIP joints (6, 10, 14, 18)
    fingers_folded = True
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        dist_tip_wrist = np.linalg.norm(pts[tip_idx] - wrist)
        dist_pip_wrist = np.linalg.norm(pts[pip_idx] - wrist)
        if dist_tip_wrist > dist_pip_wrist * 1.15:
            fingers_folded = False
            break

    if not fingers_folded:
        return False

    # 2. Check Thumb is extended UPWARDS (Thumb tip 4 is well above wrist and thumb MCP 2)
    thumb_tip = pts[4]
    thumb_ip = pts[3]
    thumb_mcp = pts[2]

    # Thumb tip must be higher up in the frame than IP joint and MCP joint (smaller Y coordinate)
    thumb_pointing_up = thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]

    # Thumb must be extended (tip is far from wrist)
    dist_thumb_wrist = np.linalg.norm(thumb_tip - wrist)
    dist_mcp_wrist = np.linalg.norm(thumb_mcp - wrist)
    thumb_extended = dist_thumb_wrist > dist_mcp_wrist * 1.3

    return thumb_pointing_up and thumb_extended
