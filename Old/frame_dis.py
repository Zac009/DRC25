import numpy as np
import cv2 as cv2
import math
from collections import defaultdict
 
cap = cv2.VideoCapture('qut_demo.mov')
# Define the codec and create VideoWriter object

center_points = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('output1.avi', fourcc, 20.0, (640,  480))
threshold1 = 85
threshold2 = 85
r_width = 500
r_height = 300
prev_gray = None
prev_points = None
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))



def get_initial_heading():
        return (0, -1)  # moving up in image space

def average_center_points(center_points):
    # Group center points by their y-value
    grouped_points = defaultdict(list)

    for x, y in center_points:
        grouped_points[y].append(x)

    # Average the x values for each unique y value
    averaged_points = []
    for y, x_vals in grouped_points.items():
        avg_x = int(np.mean(x_vals))  # Calculate the mean of x values for the same y
        averaged_points.append((avg_x, y))

    # Sort points by their y-value to maintain the path order
    averaged_points.sort(key=lambda p: p[1])  # Sort by y-value (row)

    return averaged_points

def get_perpendicular_scan(start_point, direction, length):
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

def adaptive_centerline(mask_yellow, mask_blue, num_steps=1, step_size=10):
    w, h = mask_yellow.shape
    se = w/2
    position = (h/2, se)
    direction = get_initial_heading()

    midpoint_old = None
    for _ in range(num_steps):
        scan_mask = np.zeros_like(combined_mask)
        left_pt, right_pt = get_perpendicular_scan(position, direction, length=350)
        """print(f"Left point {left_pt}")
        print(f"Right point {right_pt}")"""
        # Create scanline as a mask
        #cv2.line(scan_mask, (10,vals[1]), (490,vals[3]), 255, 1)
        cv2.line(scan_mask, left_pt, right_pt, 255, 1)
        cv2.line(combined_mask, left_pt, right_pt, 255, 1)
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
            center_points.append(midpoint)
            if midpoint_old is not None:
                dx = midpoint[0] - midpoint_old[0]
                dy = midpoint[1] - midpoint_old[1]

                # Calculate the angle in radians
                angle_rad = math.atan2(dy, dx)
                angle_rad = -angle_rad
                direction = rotate_direction(direction, angle_rad)
                se -= step_size
                position = (midpoint[0], se)
            else:
                se -= step_size
                position = (midpoint[0], se)
            midpoint_old = midpoint
            cv2.circle(combined_mask, midpoint, 3, (255, 255, 255), -1)
            #cv2.circle(combined_mask, (int(yellow_mean[0]), int(yellow_mean[1])), 3, (255, 255, 255), 5)
            #cv2.circle(combined_mask, (int(blue_mean[0]), int(blue_mean[1])), 3, (255, 255, 255), 5)

        else:
            print("Woah")

    return scan_mask


def track_frame_motion(prev, gray):
    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Calculate the total displacement of the frame
    dx, dy = np.mean(flow, axis=(0, 1))  # This gives you the average displacement
    return dx, dy

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break 
    # write the flipped frame
    #out.write(frame)
    # Convert to HSV and create masks
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 50, 120])
    upper_blue = np.array([150, 255, 255])
    blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)

    lower_yellow = np.array([25, 30, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)

    # Optional: just to visualize
    combined_mask = cv2.add(yellow_mask, blue_mask)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Optional: Gaussian blur can also help
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours on cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track frame motion and calculate displacement
    if prev_gray is not None:
        dx, dy = track_frame_motion(prev_gray, gray)
        
        # Adjust all previous center points based on the displacement
        if center_points:
            center_points_new = [(cx + dx, cy + dy) for cx, cy in center_points]
            if abs(center_points[-1][0]-center_points_new[-1][0]) > 100:
                center_points_new.pop()
            center_points = center_points_new
    edged1 = cv2.Canny(yellow_mask, threshold1, threshold2)
    edged2 = cv2.Canny(blue_mask, threshold1, threshold2)

    path_points = adaptive_centerline(yellow_mask, blue_mask)
    averaged_center_points = average_center_points(center_points)
    pts = np.array(center_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(combined_mask, [pts], isClosed=False, color=255, thickness=2)
        
    cv2.imshow('frame', combined_mask)
    """key = cv2.waitKey(0)  # 0 means "wait forever"
    if key == ord('q'):
        break  # Exit on 'q'"""
    if cv2.waitKey(1) == ord('q'):
        break
    prev_gray = gray
 
# When everything done, release the capture
cap.release()
#out.release()
cv2.destroyAllWindows()