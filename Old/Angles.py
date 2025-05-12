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
    return (0.1, -1)  # moving up in image space

def get_perpendicular_scan(start_point, direction, length=50):
    dx, dy = direction
    perp = (-dy, dx)
    left = (int(start_point[0] + perp[0] * length), int(start_point[1] + perp[1] * length))
    right = (int(start_point[0] - perp[0] * length), int(start_point[1] - perp[1] * length))
    return left, right

def rotate_direction(direction, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    dx, dy = direction

    # Rotation matrix
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    dx_rot = dx * cos_theta - dy * sin_theta
    dy_rot = dx * sin_theta + dy * cos_theta

    return (dx_rot, dy_rot)

def adaptive_centerline(mask_yellow, mask_blue, num_steps=2, step_size=10):
    h, w = mask_yellow.shape
    center_points = []

    position = (250, 250)
    direction = get_initial_heading()

    vals = [10,290,490,290]
    scan_mask = np.zeros_like(mask_yellow)
    for _ in range(num_steps):
        left_pt, right_pt = get_perpendicular_scan(position, direction, length=200)
        print(f"Left point {left_pt}")
        print(f"Right point {right_pt}")
        # Create scanline as a mask
        #cv2.line(scan_mask, (10,vals[1]), (490,vals[3]), 255, 1)
        cv2.line(scan_mask, left_pt, right_pt, 255, 1)
        # Mask and get pixel hits

        cv2.imshow("Adaptive Pathing", scan_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

path_points = adaptive_centerline(yellow_mask, blue_mask)
