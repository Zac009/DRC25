import numpy as np
import cv2 as cv2
import math
import time
import pigpio
"""
Ideas to fix:
*Take the average of say 10 points and fin the line of best fit
*Move the points more to be more spaced out.
*Fix mask to be more accurate with less noise
"""

# GPIO pin setup
STEER_PIN = 12
DRIVE_PIN = 16

# Servo pulse values
STEER_LEFT = 1000
STEER_CENTER = 1500
STEER_RIGHT = 2000

DRIVE_STOP = 1500
DRIVE_FORWARD = 1600
DRIVE_BACKWARD = 1400

frame_count = 0
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object

center_points = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = cap.read()
height, width = frame.shape[:2]
out = cv2.VideoWriter('output20.avi', fourcc, 20.0, (width, height))
threshold1 = 85
threshold2 = 85
r_width = 500
r_height = 300
prev_gray = None
prev_points = None

def steer(pulse):
    pi.set_servo_pulsewidth(STEER_PIN, pulse)

def drive(pulse):
    pi.set_servo_pulsewidth(DRIVE_PIN, pulse)

def movement(num):
    pass

def yellow_det(mask):
    lower_yellow = np.array([25,50,100]) #Was 30
    upper_yellow = np.array([50,255,255])
    yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
    return yellow_mask

def blue_det(mask):
    lower_blue = np.array([100,50,120])
    upper_blue = np.array([150,255,255])
    blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)
    return blue_mask

def calculate_angle(mid):
    temp1 = width/2
    w = abs(mid[0]-temp1)
    temp2 = height
    h = abs(mid[1]-temp2)
    """print(f"This is the height: {mid[0]}, and this is the width {mid[1]}")
    print(f"This is the height: {temp1}, and this is the width {temp2}")
    print(f"This is the result: {h}, and this is the result {w}")"""
    try:
        ang = math.atan(h/w)
        ang = math.degrees(ang)
    except ZeroDivisionError:
        ang = 90
    if mid[0] > width/2:
        ang = 180-ang
    pulse = 1000 + (ang / 180.0) * 1000
    return pulse

def check_green(mid):
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 70, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(frame_HSV, lower_green, upper_green)
    final_mask = np.zeros_like(green_mask)
    cv2.circle(final_mask, mid, 3, (255, 255, 255), -1)
    new_new = np.bitwise_and(green_mask, final_mask)
    if cv2.countNonZero(new_new) != 0:
        print("Finish line is in sight")
    drive(DRIVE_STOP)
    #Stop code

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
        #detect_box(mix_mask)
        if yellow_coords is not None and blue_coords is not None:
            yellow_mean = np.mean(yellow_coords, axis=0)[0]
            blue_mean = np.mean(blue_coords, axis=0)[0]
            # Midpoint
            midpoint_x = int((yellow_mean[0] + blue_mean[0]) / 2)
            midpoint_y = int((yellow_mean[1] + blue_mean[1]) / 2)
            midpoint = (midpoint_x, midpoint_y)
            print(midpoint)
            #check_green(midpoint)
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
    ang = calculate_angle(midpoint)
    movement(ang)
    return scan_mask, mix_mask, ang

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
    return final

pi = pigpio.pi()
if not pi.connected:
    print("Pi is not running")
    exit()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

pi.set_servo_pulsewidth(DRIVE_PIN, 1520) 
try:
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break 
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = yellow_det(frame_HSV)
        blue_mask = blue_det(frame_HSV)

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
        path_points, mask3, pulse_value = adaptive_centerline(yellow_mask, blue_mask)
        pts = np.array(center_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(combined_mask, [pts], isClosed=False, color=255, thickness=2)
        frame_count += 1
        pi.set_servo_pulsewidth(STEER_PIN, pulse_value)
        time.sleep(0.05)
        out.write(combined_mask)
        cv2.imshow('FINAL', combined_mask)
        """cv2.imshow('frame', mask3)
        cv2.moveWindow("frame", 700, 0)"""
        cv2.imshow('frame2', frame)
        cv2.moveWindow("frame2", 700, 500)
        """cv2.imshow('frame3', scan_mask)
        cv2.moveWindow("frame3", 0, 500)"""
        if cv2.waitKey(1) == ord('q'):
            break
        prev_gray = gray
finally:
    # Cleanup
    drive(DRIVE_STOP)
    steer(STEER_CENTER)
    pi.set_servo_pulsewidth(DRIVE_PIN, 0)
    pi.set_servo_pulsewidth(STEER_PIN, 0)
    pi.stop()
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
