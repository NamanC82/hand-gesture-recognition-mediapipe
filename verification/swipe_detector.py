import numpy as np


def is_valid_swipe(point_history, min_displacement=280, min_points=6, max_vertical_ratio=0.6):
    """Detects a fast, deliberate horizontal swipe.

    Optimized for rapid motion, frame drops, and larger swipe distances.
    """
    if not point_history:
        return False

    # Filter out [0, 0] padding points
    valid_points = [p for p in point_history if len(p) >= 2 and (p[0] != 0 or p[1] != 0)]

    if len(valid_points) < min_points:
        return False

    pts = np.array(valid_points, dtype=np.float64)

    # Calculate overall coordinate bounding box size of the trajectory
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    x_span = x_max - x_min
    y_span = y_max - y_min

    # Calculate net progression from start to end
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]

    # Check 1: The horizontal span of the swipe must be at least min_displacement pixels
    if x_span < min_displacement:
        return False

    # Check 2: The gesture should be primarily horizontal (not diagonal/vertical)
    if y_span / (x_span + 1e-6) > max_vertical_ratio:
        return False

    # Check 3: Direction consistency. Ensure start-to-end net displacement
    # accounts for at least 80% of the overall horizontal span.
    # This prevents hand-waving back and forth from triggering a swipe.
    if abs(dx) / (x_span + 1e-6) < 0.8:
        return False

    return True
