import cv2
import math

def change_detection(bbox, depth_prev, position_prev):
    """Detects changes in x, y, and depth position between object location events"""
    x1, y1, x2, y2 = bbox

    # Current values
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    depth_initial = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Differences from previous
    delta_x = abs(mx - position_prev[0])
    delta_y = abs(my - position_prev[1])
    delta_depth = abs(depth_initial - depth_prev)

    # Change thresholds
    position_threshold = 50   # pixels
    depth_threshold = 0.2 * depth_prev  # 20% change in apparent size

    significant_change = False
    if delta_x > position_threshold:
        print("Change in X")
        significant_change = True
    if delta_y > position_threshold:
        print("Change in Y")
        significant_change = True
    if delta_depth > depth_threshold:
        print("Change in depth")
        significant_change = True

    return significant_change, delta_x, delta_y, delta_depth, depth_initial

def initial_change_states(bbox):
    """Compute initial depth and position"""
    x1, y1, x2, y2 = bbox
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    depth_initial = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    position_initial = (mx, my)
    return depth_initial, position_initial


# --- Simulated frames ---
frames = [
    [0  , 0  , 0  , 0  ],
    [250, 200, 350, 300],
    [300, 200, 400, 300]
]

# Initialize
depth_initial, position_initial = initial_change_states(frames[0])
print(f"Initial: depth={depth_initial}, position={position_initial}")

# Iterate over frames
for i, bbox in enumerate(frames[1:], start=1):
    print(f"\n--- Frame {i} ---")
    significant_change, dx, dy, dd, depth_initial, position_initial = change_detection(
        bbox, depth_initial, position_initial
    )

    print(f"Significant change: {significant_change}")
    print(f"Δx={dx}, Δy={dy}, Δdepth={dd}")
    print(f"New depth={depth_initial}, New position={position_initial}")
