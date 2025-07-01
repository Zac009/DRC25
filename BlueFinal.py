import numpy as np
import cv2 as cv2
import math
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
DRIVE_FORWARD = 1680
DRIVE_CORNER = 1670
DRIVE_BACKWARD = 1400

class Vision:
    def __init__(self):
        self.frame_count = 0
        #self.center_points = []
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
        self.brake = False
        self.state = None
    
    def blue_det(self):
        lower_blue = np.array([100,50,120])
        upper_blue = np.array([150,255,255])
        blue_mask = cv2.inRange(self.frame_HSV, lower_blue, upper_blue)
        return blue_mask
    
    def green_det(self):
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(self.frame_HSV, lower_green, upper_green)
        return green_mask
    
    def steer(self,pulse):
        self.pi.set_servo_pulsewidth(STEER_PIN, pulse)

    def drive(self, pulse):
        self.pi.set_servo_pulsewidth(DRIVE_PIN, pulse)
    
    def calculate_ang(self, mid):
        temp1 = self.width/2
        w = abs(mid[0]-temp1)
        temp2 = self.frame_HSV.shape[0] 
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
    
    #Num steps is for individual photos not videos
    def adaptive_centerline(self, mask_blue, num_steps=1, step_size=10):
        w, h = mask_blue.shape
        position = (w // 2, h // 2)
        direction = self.get_initial_heading()
        midpoint_old = None
        midpoint = None
        for _ in range(num_steps):
            left_pt, right_pt = self.get_perpendicular_scan(position, direction, length=400)
            # Create scanline as a mask
            #cv2.line(scan_mask, (10,vals[1]), (490,vals[3]), 255, 1)
            cv2.line(self.scan_mask, left_pt, right_pt, 255, 1)
            #cv2.line(self.combined_mask, left_pt, right_pt, 255, 1)
            # Mask and get pixel hits
            blue_hits = cv2.bitwise_and(mask_blue, self.scan_mask)
            green_mask = self.green_det()
            green_hits = cv2.bitwise_and(green_mask, self.scan_mask)
            mix_mask = blue_hits
            blue_coords = cv2.findNonZero(blue_hits)
            green_coords = cv2.findNonZero(green_hits)
            #one, self.two = self.detect_box(mix_mask)
            if green_coords is not None:
                ang = 0
                print("End Line Found")
            elif blue_coords is not None:
                blue_mean = np.mean(blue_coords, axis=0)[0]
                print("Blue Coords found")
                #midpoint_x = int(blue_mean[0]) - (self.width // 4)
                midpoint_x = int(blue_mean[0]) - 50
                midpoint_y = left_pt[1]
                midpoint = (midpoint_x, midpoint_y)
                #self.center_points.append(midpoint)
                #cv2.circle(self.combined_mask, midpoint, 3, (255, 255, 255), -1)
                ang = self.calculate_ang(midpoint)
            else:
                if self.brake == False:
                    print("No colors found!!!")
                    var = 2000 - self.prev_pulse
                    var /= 100
                    var = int(var)
                    print(f"This is the variable: {var}")
                    self.drive(DRIVE_CORNER)
                    time.sleep(0.6)
                    for i in range(10):
                        ret, self.frame = self.cap.read()
                        self.frame_HSV = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                        self.steer(1900)
                        self.drive(DRIVE_CORNER)
                        time.sleep(0.3)
                        blue_mask = self.blue_det()
                        blue_hits = cv2.bitwise_and(blue_mask, self.scan_mask)
                        blue_coords = cv2.findNonZero(blue_hits)
                        mix_mask = blue_hits
                        if blue_coords is not None:
                            blue_mean = np.mean(blue_coords, axis=0)[0]
                            print("boom\n")
                            #midpoint_x = int(blue_mean[0]) - (self.width // 4)
                            midpoint_x = int(blue_mean[0]) - 50
                            midpoint_y = left_pt[1]
                            midpoint = (midpoint_x, midpoint_y)
                            #self.center_points.append(midpoint)
                            ang = self.calculate_ang(midpoint)
                            if ang >= 1650 or ang <= 1450:
                                self.drive(DRIVE_CORNER)
                                time.sleep(1)
                            break
                else:
                    ang = 0
        return self.scan_mask, mix_mask, ang
    
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
                #self.center_points.append(midpoint)
                
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
        #self.__init__(self.lock)
        cap = cv2.VideoCapture(0)
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.running = True
        self.pi = pigpio.pi()
        # Get first valid frame
        ret, self.frame = cap.read()
        if not ret:
            print("Can't receive initial frame. Exiting ...")
            return
        self.height, self.width = self.frame.shape[:2]
        if not self.pi.connected:
            print("Pi is not running")
            exit()
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        #self.steer(STEER_CENTER)
        try:
            while True:
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break 
                #self.frame_HSV = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                roi_top = self.height // 3        # 1/3 down from top
                roi_bottom = 2 * self.height // 3 # 2/3 down from top
                roi = self.frame[roi_top:roi_bottom, :]  # All columns

                # Store for later use
                self.roi = roi
                self.frame_HSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                self.height = self.frame_HSV.shape[0]
                self.width = self.frame_HSV.shape[1]
                blue_mask = self.blue_det()

                # Optional: just to visualize
                #self.combined_mask = np.zeros_like(self.frame)
                #self.combined_mask = cv2.add(yellow_mask, blue_mask)
                self.scan_mask = np.zeros_like(self.frame_HSV[:, :, 0]) 
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                path_points, mask3, ang = self.adaptive_centerline(blue_mask)
                """pts = np.array(self.center_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.combined_mask, [pts], isClosed=False, color=255, thickness=2)"""
                self.frame_count += 1
                print(ang)
                try:
                    if ang == 0:
                        self.drive(DRIVE_STOP)
                        self.running = False
                    else:
                        if abs(self.prev_pulse - ang) > 30:
                            self.steer(ang)
                            self.prev_pulse = ang
                            if self.state == "CORRRNER":
                                pass
                            else:
                                print("Corner")
                                self.drive(DRIVE_CORNER)
                                self.state = "CORNER"
                            time.sleep(0.5)
                            self.drive(DRIVE_STOP)
                        else:
                            if self.state == "FORWARRRD":
                                pass
                            else:
                                print("Forward")
                                self.drive(DRIVE_FORWARD)
                                self.state = "FORWARD"
                            time.sleep(0.5)
                            self.drive(DRIVE_STOP)
                except Exception as e:
                    print(f"There was an error: {e}")
                    self.drive(DRIVE_STOP)
                #self.combined_mask = cv2.bitwise_or(self.two, self.combined_mask)
                if not self.running:
                    print("Process Stopped...")
                    break
                self.prev_gray = gray
        except KeyboardInterrupt:
                print("Interrupted by user")
                self.running = False
                self.drive(DRIVE_STOP)
        finally:
            # When everything done, release the capture
            print("FINISHED")
            self.drive(DRIVE_STOP)
            self.pi.stop()
            cap.release()
            cv2.destroyAllWindows()


Ben = Vision()
Ben.main()
