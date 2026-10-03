import numpy as np


def is_valid_move(palm_history, min_displacement=100, min_points=8):
    """Detects whole-hand movement by tracking the palm/wrist center.

    The entire hand must move significantly, not just a finger.

    palm_history: list of [x, y] positions of the palm center (landmark 9)
                  or wrist (landmark 0) over recent frames.
    """
    if not palm_history:
        return False

    valid = [p for p in palm_history if len(p) >= 2 and (p[0] != 0 or p[1] != 0)]
    if len(valid) < min_points:
        return False

    pts = np.array(valid, dtype=np.float64)

    # Total displacement of the palm from start to end
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    total_displacement = np.sqrt(dx**2 + dy**2)

    if total_displacement < min_displacement:
        return False

    # Check that movement is somewhat continuous (not just a jump)
    diffs = np.diff(pts, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    avg_step = step_distances.mean()
    if avg_step < 3.0:
        return False  # Hand barely moved between frames

    return True
