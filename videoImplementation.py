import numpy as np
import cv2 as cv2
import math

"""
Ideas to fix:
*Take the average of say 10 points and fin the line of best fit
*Move the points more to be more spaced out.
*Fix mask to be more accurate with less noise
"""

frame_count = 0
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


def get_initial_heading():
        return (0, -1)  # moving up in image space

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
        mix_mask = cv2.bitwise_or(yellow_hits, blue_hits)
        yellow_coords = cv2.findNonZero(yellow_hits)
        blue_coords = cv2.findNonZero(blue_hits)
        detect_box(mix_mask)
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
                norm = math.hypot(dx, dy)
                if norm != 0:
                    direction = (dx / norm, dy / norm)
                se -= step_size
                position = (midpoint[0], se)
            else:
                se -= step_size
                position = (midpoint[0], se)
            midpoint_old = midpoint
            cv2.circle(combined_mask, midpoint, 3, (255, 255, 255), -1)
            cv2.circle(mix_mask, midpoint, 3, (255, 255, 255), -1)
            #cv2.circle(combined_mask, (int(yellow_mean[0]), int(yellow_mean[1])), 3, (255, 255, 255), 5)
            #cv2.circle(combined_mask, (int(blue_mean[0]), int(blue_mean[1])), 3, (255, 255, 255), 5)

        else:
            pass
    """cv2.imshow('frame', mix_mask)
    cv2.moveWindow("frame", 700, 0)"""
    cv2.imshow('frame2', frame)
    cv2.moveWindow("frame2", 700, 500)
    return scan_mask

def track_frame_motion(prev, gray):
    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Calculate the total displacement of the frame
    dx, dy = np.mean(flow, axis=(0, 1)) 
    dx*=5
    dy*=5 # This gives you the average displacement
    return dx, dy

def detect_box(hit_mask):
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 70, 70])   # H, S, V
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(frame_HSV, lower_purple, upper_purple)
    contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(purple_mask)
    cv2.drawContours(final, contours, -1, (255, 255, 255), 2)
    new_final = cv2.bitwise_and(scan_mask, final)
    final_final = cv2.bitwise_or(new_final, hit_mask)
    all_coords = cv2.findNonZero(final_final)
    final = np.zeros_like(purple_mask)
    if all_coords is not None and len(all_coords) >= 20:
     # Flatten and sort points left to right
        sorted_coords = sorted([pt[0] for pt in all_coords], key=lambda p: p[0])  # Sort by x
    
        # Track the maximum gap and its index
        max_gap = 0
        max_pair = None
        
        for i in range(len(sorted_coords) - 1):
            p1 = sorted_coords[i]
            p2 = sorted_coords[i + 1]
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            if dist > max_gap:
                max_gap = dist
                max_pair = (p1, p2)
        
        if max_pair:
            midpoint_x = int((max_pair[0][0] + max_pair[1][0]) / 2)
            midpoint_y = int((max_pair[0][1] + max_pair[1][1]) / 2)
            midpoint = (midpoint_x, midpoint_y)
            center_points.append(midpoint)
            
            # Visualize
            cv2.circle(final, midpoint, 4, (255, 255, 255), -1)
            final = cv2.bitwise_or(final,final_final)
    cv2.imshow('scan', final)
    cv2.moveWindow("scan", 0, 500)
    cv2.imshow('frame', purple_mask)
    cv2.moveWindow("frame", 700, 0)
    return final

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

    lower_blue = np.array([100,50,120])
    upper_blue = np.array([150,255,255])
    lower_yellow = np.array([30,50,100])
    upper_yellow = np.array([50,255,255])
    yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)

    # Optional: just to visualize
    combined_mask = cv2.add(yellow_mask, blue_mask)
    scan_mask = np.zeros_like(combined_mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track frame motion and calculate displacement
    if prev_gray is not None:
        dx, dy = track_frame_motion(prev_gray, gray)
        center_points = [(cx + dx, cy + dy) for cx, cy in center_points]

    yellow_hits = cv2.bitwise_and(yellow_mask, scan_mask)
    blue_hits = cv2.bitwise_and(blue_mask, scan_mask)
    path_points = adaptive_centerline(yellow_mask, blue_mask)
    pts = np.array(center_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(combined_mask, [pts], isClosed=False, color=255, thickness=2)
    frame_count += 1
    cv2.imshow('FINAL', combined_mask)
    if cv2.waitKey(1) == ord('q'):
        break
    prev_gray = gray
 
# When everything done, release the capture
cap.release()
#out.release()
cv2.destroyAllWindows()