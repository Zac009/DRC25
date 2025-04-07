import cv2
import numpy as np
import math

threshold1 = 85
threshold2 = 85
r_width = 500
r_height = 300

image = cv2.imread(r'test2.png')
image = cv2.resize(image, (r_width, r_height))

# Convert to HSV and create masks
frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 50, 120])
upper_blue = np.array([150, 255, 255])
blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)

lower_yellow = np.array([25, 30, 100])
upper_yellow = np.array([40, 255, 255])
yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)

# Optional: just to visualize
combined_mask = cv2.add(yellow_mask, blue_mask)
edged1 = cv2.Canny(yellow_mask, threshold1, threshold2)
edged2 = cv2.Canny(blue_mask, threshold1, threshold2)

def get_initial_heading():
    return (0, -1)  # moving up in image space

def get_perpendicular_scan(start_point, direction, length=50):
    dx, dy = direction
    perp = (-dy, dx)
    left = (int(start_point[0] + perp[0] * length), int(start_point[1] + perp[1] * length))
    right = (int(start_point[0] - perp[0] * length), int(start_point[1] - perp[1] * length))
    return left, right

def adaptive_centerline(mask_yellow, mask_blue, num_steps=28, step_size=10):
    h, w = mask_yellow.shape
    center_points = []

    position = (250, 250)
    direction = get_initial_heading()

    vals = [10,290,490,290]
    scan_mask = np.zeros_like(mask_yellow)
    previous_midpoint = position  # Initialize previous midpoint to start position
    
    for _ in range(num_steps):
        left_pt, right_pt = get_perpendicular_scan(position, direction, length=200)
        # Create scanline as a mask
        cv2.line(scan_mask, left_pt, right_pt, 255, 1)
        cv2.imshow("Adaptive Pathing", scan_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Mask and get pixel hits
        yellow_hits = cv2.bitwise_and(mask_yellow, scan_mask)
        blue_hits = cv2.bitwise_and(mask_blue, scan_mask)

        yellow_coords = cv2.findNonZero(yellow_hits)
        blue_coords = cv2.findNonZero(blue_hits)
        if yellow_coords is not None and blue_coords is not None:
            yellow_mean = np.mean(yellow_coords, axis=0)[0]
            blue_mean = np.mean(blue_coords, axis=0)[0]

            # Midpoint
            midpoint_x = int((yellow_mean[0] + blue_mean[0]) / 2)
            midpoint_y = int((yellow_mean[1] + blue_mean[1]) / 2)
            midpoint = (midpoint_x, midpoint_y)
            cv2.circle(combined_mask, midpoint, 3, (255, 255, 255), -1)

            # Update heading based on previous midpoint
            dx = midpoint[0] - previous_midpoint[0]
            dy = midpoint[1] - previous_midpoint[1]
            norm = math.hypot(dx, dy)
            if norm > 0:
                direction = (dx / norm, dy / norm)

            # Move along the updated heading (parallel direction)
            position = (
                int(midpoint[0] + direction[0] * step_size),
                int(midpoint[1] + direction[1] * step_size)
            )
            previous_midpoint = midpoint  # Update previous midpoint
        else:
            print("Woah")

        vals[1] -= step_size
        vals[3] -= step_size


path_points = adaptive_centerline(yellow_mask, blue_mask)
