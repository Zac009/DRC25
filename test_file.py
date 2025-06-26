import numpy as np
import cv2 as cv2
import math
import time

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
        self.prev_pulse = 1550
        self.running = False
        self.mask1 = None
        self.mask2 = None 
        self.lock = lock

    def yellow_det(self, mask):
        lower_yellow = np.array([26,50,100])
        upper_yellow = np.array([50,255,255])
        yellow_mask = cv2.inRange(self.frame_HSV, lower_yellow, upper_yellow)
        return yellow_mask
    
    def blue_det(self, mask):
        lower_blue = np.array([100,50,120])
        upper_blue = np.array([150,255,255])
        blue_mask = cv2.inRange(self.frame_HSV, lower_blue, upper_blue)
        return blue_mask
    
    def green_det(self, mid):
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(self.frame_HSV, lower_green, upper_green)
        #Stop Code
        return green_mask
    
    def get_initial_heading(self):
        return (0, -1)
    
    def get_perpendicular_scan(self, start_point, direction, length):
        dx, dy = direction
        perp = (-dy, dx)
        left = (int(start_point[0] + perp[0] * length), int(start_point[1] + perp[1] * length))
        right = (int(start_point[0] - perp[0] * length), int(start_point[1] - perp[1] * length))
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
    def adaptive_centerline(self, mask_yellow, mask_blue, num_steps=1, step_size=10):
        w, h = mask_yellow.shape
        se = w/2
        position = (h/2, se)
        direction = self.get_initial_heading()

        midpoint_old = None
        midpoint = None
        for _ in range(num_steps):
            left_pt, right_pt = self.get_perpendicular_scan(position, direction, length=350)
            cv2.line(self.scan_mask, left_pt, right_pt, 255, 1)
            cv2.line(self.combined_mask, left_pt, right_pt, 255, 1)
            
    
    def main(self):
        self.__init__(self.lock)
        cap = cv2.VideoCapture(0)
        self.running = True
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, self.frame = cap.read()
            self.height, self.width = self.frame.shape[:2]
        
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break 
            self.frame_HSV = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            yellow_mask = self.yellow_det(self.frame_HSV)
            blue_mask = self.blue_det(self.frame_HSV)

            # Optional: just to visualize
            self.combined_mask = np.zeros_like(self.frame)
            self.combined_mask = cv2.add(yellow_mask, blue_mask)
            self.scan_mask = np.zeros_like(self.combined_mask)
            self.adaptive_centerline(yellow_mask, blue_mask)
            with self.lock:
                self.mask1 = blue_mask
                self.mask2 = self.frame
            if not self.running:
                print("Process Stopped...")
                break
        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
