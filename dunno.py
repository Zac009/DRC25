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

def rotate_direction(direction, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    dx, dy = direction

    # Rotation matrix
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    dx_rot = dx * cos_theta - dy * sin_theta
    dy_rot = dx * sin_theta + dy * cos_theta

    return (dx_rot, dy_rot)

def adaptive_centerline(mask_yellow, mask_blue, num_steps=10000, step_size=5):
    h, w = mask_yellow.shape
    center_points = []
    se = 250
    position = (250, se)
    direction = get_initial_heading()

    vals = [10, 290, 490, 290]
    scan_mask = np.zeros_like(mask_yellow)
    midpoint_old = None

    for _ in range(num_steps):
        # Get the current midpoint (which we want to track)
        left_pt, right_pt = get_perpendicular_scan(position, direction, length=200)

        # Calculate the midpoint as the center of the line
        left_pt_x, left_pt_y = left_pt
        right_pt_x, right_pt_y = right_pt
        midpoint_x = (left_pt_x + right_pt_x) // 2
        midpoint_y = (left_pt_y + right_pt_y) // 2
        midpoint = (midpoint_x, midpoint_y)

        # Use the midpoint to update the scanline and direction
        if midpoint_old is not None:
            dx = midpoint_x - midpoint_old[0]
            dy = midpoint_y - midpoint_old[1]

            # Calculate the angle in radians
            angle_rad = math.atan2(dy, dx)
            angle_rad = -angle_rad  # Flip angle to match your coordinate system
            direction = rotate_direction(direction, angle_rad)
            position = (midpoint[0], midpoint_y)  # Update position based on the midpoint

        # Update the scanline on the scan_mask
        cv2.line(scan_mask, left_pt, right_pt, 255, 1)

        # Check for intersections with the yellow and blue masks
        yellow_hits = cv2.bitwise_and(mask_yellow, scan_mask)
        blue_hits = cv2.bitwise_and(mask_blue, scan_mask)

        yellow_coords = cv2.findNonZero(yellow_hits)
        blue_coords = cv2.findNonZero(blue_hits)

        if yellow_coords is not None and blue_coords is not None:
            yellow_mean = np.mean(yellow_coords, axis=0)[0]
            blue_mean = np.mean(blue_coords, axis=0)[0]

            # Update the midpoint
            midpoint_x = int((yellow_mean[0] + blue_mean[0]) / 2)
            midpoint_y = int((yellow_mean[1] + blue_mean[1]) / 2)
            midpoint = (midpoint_x, midpoint_y)
            cv2.circle(combined_mask, midpoint, 3, (255, 255, 255), -1)

        # Move the scanning line down (based on your step_size)
        se -= step_size

        # Visualize the scanmask and combined mask
        cv2.namedWindow("Window 2", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Window 2", 500, 500)
        cv2.imshow("Window 2", scan_mask)
        cv2.imshow("Combined Pathing", combined_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        midpoint_old = midpoint

path_points = adaptive_centerline(yellow_mask, blue_mask)


path_points = adaptive_centerline(yellow_mask, blue_mask)
