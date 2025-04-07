import cv2
import numpy as np
import math

# ======== CONFIG ==========
threshold1 = 85
threshold2 = 85
r_width = 500
r_height = 300

# ======== IMAGE LOAD & PREP ==========
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

# ======== ADAPTIVE SCANLINE CENTERLINE ==========
def get_initial_heading():
    return (0, -1)  # moving up in image space

def get_perpendicular_scan(start_point, direction, length=50):
    dx, dy = direction
    perp = (-dy, dx)
    left = (int(start_point[0] + perp[0] * length), int(start_point[1] + perp[1] * length))
    right = (int(start_point[0] - perp[0] * length), int(start_point[1] - perp[1] * length))
    return left, right

def adaptive_centerline(mask_yellow, mask_blue, num_steps=30, step_size=10):
    h, w = mask_yellow.shape
    center_points = []

    position = (w // 2, h - 10)
    print(position)
    """direction = get_initial_heading()

    for _ in range(num_steps):
        left_pt, right_pt = get_perpendicular_scan(position, direction, length=30)

        # Create scanline as a mask
        scan_mask = np.zeros_like(mask_yellow)
        cv2.line(scan_mask, left_pt, right_pt, 255, 1)

        # Mask and get pixel hits
        yellow_hits = cv2.bitwise_and(mask_yellow, scan_mask)
        blue_hits = cv2.bitwise_and(mask_blue, scan_mask)

        yellow_coords = cv2.findNonZero(yellow_hits)
        blue_coords = cv2.findNonZero(blue_hits)

        if yellow_coords is not None and blue_coords is not None:
            left = tuple(yellow_coords.min(axis=0)[0])
            right = tuple(blue_coords.max(axis=0)[0])
            center_x = (left[0] + right[0]) // 2
            center_y = (left[1] + right[1]) // 2
            center = (center_x, center_y)
            center_points.append(center)

            # Update direction and position
            dx = center[0] - position[0]
            dy = center[1] - position[1]
            norm = math.hypot(dx, dy)
            if norm > 1:
                direction = (dx / norm, dy / norm)
            position = center
        else:
            break  # Stop if we lose track

    return center_points

def draw_path(img, points):
    if not points:
        return img

    # Draw points
    for pt in points:
        cv2.circle(img, pt, 3, (0, 255, 0), -1)

    # Draw lines between points
    for i in range(1, len(points)):
        cv2.line(img, points[i - 1], points[i], (0, 255, 0), 2)

    return img

# ======== RUN PATHING + DISPLAY ==========
#output = draw_path(image.copy(), path_points)
print(path_points)
cv2.imshow("Adaptive Pathing", output)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
path_points = adaptive_centerline(yellow_mask, blue_mask)
