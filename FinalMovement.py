import numpy as np
import cv2 as cv2
import math
from math import atan2, cos, sin, sqrt, pi
import time
import pigpio

# GPIO pin setup
STEER_PIN = 12
DRIVE_PIN = 16

# Servo pulse values
STEER_LEFT = 1000
STEER_CENTER = 1500
STEER_RIGHT = 2000

DRIVE_STOP = 1500
DRIVE_FORWARD = 1600
DRIVE_CORNER = 1590
DRIVE_BACKWARD = 1400

class Vision:
    def __init__(self, lock):
        self.frame_count = 0
        self.center_points = []
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        """self.height, self.width = self.frame.shape[:2]
        self.out = cv2.VideoWriter('output20.avi', self.fourcc, 20.0, (self.width, self.height))"""
        self.threshold1 = 85
        self.threshold2 = 85
        self.r_width = 500
        self.r_height = 300
        self.prev_gray = None
        self.prev_points = None
        self.prev_pulse = 1500
        self.running = False
        self.mask1 = None
        self.mask2 = None 
        self.lock = lock
        self.state = None

    def yellow_det(self):
        lower_yellow = np.array([25,50,100])
        upper_yellow = np.array([50,255,255])
        yellow_mask = cv2.inRange(self.frame_HSV, lower_yellow, upper_yellow)
        return yellow_mask
    
    def blue_det(self):
        lower_blue = np.array([100,50,120])
        upper_blue = np.array([150,255,255])
        blue_mask = cv2.inRange(self.frame_HSV, lower_blue, upper_blue)
        return blue_mask
    
    def green_det(self):
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(self.frame_HSV, lower_green, upper_green)
        self.drive(DRIVE_STOP)
        return green_mask
    
    def steer(self,pulse):
        self.pi.set_servo_pulsewidth(STEER_PIN, pulse)

    def drive(self, pulse):
        self.pi.set_servo_pulsewidth(DRIVE_PIN, pulse)
    
    def calculate_ang(self, mid):
        temp1 = self.width/2
        w = abs(mid[0]-temp1)
        temp2 = self.height
        h = abs(mid[1]-temp2)
        try:
            ang = math.atan(h/w)
            ang = math.degrees(ang)
        except ZeroDivisionError:
            ang = 90 #Needs to = 1550 because it is straight
        if mid[0] > self.width/2:
            ang = 180 - ang
        pulse = 1000 + (ang / 180) * 1000 + 50
        return pulse
    
    def get_initial_heading(self):
        return (0, -1)
    
    def get_perpendicular_scan(self, start_point, direction, length):
        dx, dy = direction
        perp = (-dy, dx)
        left = (int(start_point[0] + perp[0] * length), int(start_point[1] + perp[1] * length))
        right = (int(start_point[0] - perp[0] * length), int(start_point[1] - perp[1] * length))
        #To change the height change the second value of the actual left and right points
        return left, right
    
    def rotate_direction(self, direction, angle_degrees):
        angle_rad = math.radians(angle_degrees)
        dx, dy = direction

        # Rotation matrix
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        dx_rot = dx * cos_theta - dy * sin_theta
        dy_rot = dx * sin_theta + dy * cos_theta

        return (dx_rot, dy_rot)
    
    def followLine(self, var):
        max_error = self.width // 2
        pulse = STEER_CENTER + int((450 * var) / max_error)
        return pulse
    
    #Num steps is for individual photos not videos
    def adaptive_centerline(self, mask_yellow, mask_blue, num_steps=1, step_size=10):
        w, h = mask_yellow.shape
        se = w/2
        position = (h/2, se) #If I need to change the positioning of the scan line
        direction = self.get_initial_heading()
        midpoint_old = None
        midpoint = None
        for _ in range(num_steps):
            left_pt, right_pt = self.get_perpendicular_scan(position, direction, length=10000)
            # Create scanline as a mask
            #cv2.line(scan_mask, (10,vals[1]), (490,vals[3]), 255, 1)
            cv2.line(self.scan_mask, left_pt, right_pt, 255, 1)
            cv2.line(self.combined_mask, left_pt, right_pt, 255, 1)
            # Mask and get pixel hits
            yellow_hits = cv2.bitwise_and(mask_yellow, self.scan_mask)
            blue_hits = cv2.bitwise_and(mask_blue, self.scan_mask)
            mix_mask = cv2.bitwise_or(yellow_hits, blue_hits)
            yellow_coords = cv2.findNonZero(yellow_hits)
            blue_coords = cv2.findNonZero(blue_hits)
            one, self.two = self.detect_box(mix_mask)
            if yellow_coords is not None and blue_coords is not None:
                yellow_mean = np.mean(yellow_coords, axis=0)[0]
                blue_mean = np.mean(blue_coords, axis=0)[0]
                # Midpoint
                midpoint_x = int((yellow_mean[0] + blue_mean[0]) / 2)
                midpoint_y = int((yellow_mean[1] + blue_mean[1]) / 2)
                midpoint = (midpoint_x, midpoint_y)
                #self.green_det(midpoint)
                self.center_points.append(midpoint)
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
                cv2.circle(self.combined_mask, midpoint, 3, (255, 255, 255), -1)
                cv2.circle(mix_mask, midpoint, 3, (255, 255, 255), -1)
                #cv2.circle(combined_mask, (int(yellow_mean[0]), int(yellow_mean[1])), 3, (255, 255, 255), 5)
                #cv2.circle(combined_mask, (int(blue_mean[0]), int(blue_mean[1])), 3, (255, 255, 255), 5)
                if midpoint is None:
                    ang = STEER_CENTER
                    print("Center steer!!!")
                else:
                    ang = self.calculate_ang(midpoint)
                    print("Midpoint Found!!!")
            elif blue_coords is not None:
                blue_mean = np.mean(blue_coords, axis=0)[0]
                print("No yellow_coords found!!!")
                midpoint_x = int(blue_mean[0]) - 300
                midpoint_y = 180
                midpoint = (midpoint_x, midpoint_y)
                self.center_points.append(midpoint)
                cv2.circle(self.combined_mask, midpoint, 3, (255, 255, 255), -1)
                ang = self.calculate_ang(midpoint)
            elif yellow_coords is not None:
                yellow_mean = np.mean(yellow_coords, axis=0)[0]
                print("No blue_coords found!!!")
                midpoint_x = int(yellow_mean[0]) + 100
                midpoint_y = 180
                midpoint = (midpoint_x, midpoint_y)
                self.center_points.append(midpoint)
                cv2.circle(self.combined_mask, midpoint, 3, (255, 255, 255), -1)
                ang = self.calculate_ang(midpoint)
                #180
            else:
                ang = 0
                print("No colors found!!!")
        return self.scan_mask, mix_mask, ang

    def track_frame_motion(self, prev, gray):
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = np.mean(flow, axis=(0,1))
        dx*=5
        dy*=5
        return dx, dy
    
    def detect_box(self, hit_mask):
        frame_HSV = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([120, 70, 70])   # H, S, V
        upper_purple = np.array([160, 255, 255])
        purple_mask = cv2.inRange(frame_HSV, lower_purple, upper_purple)
        contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final = np.zeros_like(purple_mask)
        cv2.drawContours(final, contours, -1, (255, 255, 255), 2)
        new_final = cv2.bitwise_and(self.scan_mask, final)
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
                self.center_points.append(midpoint)
                
                # Visualize
                cv2.circle(final, midpoint, 4, (255, 255, 255), -1)
                final = cv2.bitwise_or(final,final_final)
        return final, purple_mask
    
    def arrow_detection(self):
        """
        Copy the code from the arrow file
        Adjust it to make it so it only runs on a black detected color mask
        Then pass left or Right as servo pulses so 1000 or 2000
        """
        pass
    
    def main(self):
        self.__init__(self.lock)
        cap = cv2.VideoCapture(0)
        self.cap = cap
        self.running = True
        self.pi = pigpio.pi()
        if not self.pi.connected:
            print("Pi is not running")
            exit()
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        #self.steer(STEER_CENTER)
        try:
            while True:
                print("Looping")
                ret, self.frame = cap.read()
                self.height, self.width = self.frame.shape[:2]
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break 
                self.frame_HSV = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                yellow_mask = self.yellow_det()
                blue_mask = self.blue_det()

                # Optional: just to visualize
                self.combined_mask = np.zeros_like(self.frame)
                self.combined_mask = cv2.add(yellow_mask, blue_mask)
                self.scan_mask = np.zeros_like(self.combined_mask)
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # Track frame motion and calculate displacement
                if self.prev_gray is not None:
                    dx, dy = self.track_frame_motion(self.prev_gray, gray)
                    self.center_points = [(cx + dx, cy + dy) for cx, cy in self.center_points]

                yellow_hits = cv2.bitwise_and(yellow_mask, self.scan_mask)
                blue_hits = cv2.bitwise_and(blue_mask, self.scan_mask)
                path_points, mask3, ang = self.adaptive_centerline(yellow_mask, blue_mask)
                """pts = np.array(self.center_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.combined_mask, [pts], isClosed=False, color=255, thickness=2)"""
                self.frame_count += 1
                #self.out.write(self.combined_mask)
                print(ang)
                try:
                    if ang == 0:
                        self.running = False
                    else:
                        if abs(self.prev_pulse - ang) > 30:
                            self.steer(ang)
                            self.prev_pulse = ang
                            if self.state == "CORNER":
                                pass
                            else:
                                self.drive(DRIVE_CORNER)
                            time.sleep(0.01)
                        else:
                            if self.state == "FORWARD":
                                pass
                            else:
                                self.drive(DRIVE_FORWARD)
                                self.state = "FORWARD"
                            time.sleep(0.01)
                except Exception as e:
                        print(f"There was an error: {e}")
                        self.drive(DRIVE_STOP)
                #self.combined_mask = cv2.bitwise_or(self.two, self.combined_mask)
                """with self.lock:
                    self.mask1 = self.combined_mask
                    self.mask2 = self.frame"""
                """cv2.imshow('FINAL', self.combined_mask)
                cv2.imshow('frame', mask3)
                cv2.moveWindow("frame", 700, 0)
                purple_mask = self.detect_box(self.frame)
                cv2.imshow('frame2', self.frame)
                cv2.imshow('frame3', self.combined_mask)
                cv2.moveWindow("frame3", 0, 500)
                if cv2.waitKey(1) == ord('q'): #For non webserver testing.
                    break"""
                if not self.running:
                    print("Process Stopped...")
                    break
                self.prev_gray = gray
        finally:
            # When everything done, release the capture
            print("FINISHED")
            self.drive(DRIVE_STOP)
            self.pi.stop()
            cap.release()
            cv2.destroyAllWindows()


Ben = Vision("BLEH")
Ben.main()
