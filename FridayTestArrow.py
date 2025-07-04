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
DRIVE_CORNER = 1650
DRIVE_BACKWARD = 1400

DIST_FROM = 150
threshold1 = 350
threshold2 = 350

MIN_CONTOUR_AREA = 200  # Tune this value

class Vision:
    def __init__(self):
        self.r_width = 500
        self.r_height = 300
        self.direction = "Blue"
        self.left_region = (0, self.r_width // 2)
        self.right_region = (self.r_width // 2, self.r_width)

    def blue_det(self):
        lower_blue = np.array([100,50,120])
        upper_blue = np.array([150,255,255])
        blue_mask = cv2.inRange(self.frame_HSV, lower_blue, upper_blue)
        return blue_mask
    
    def yellow_det(self):
        lower_yellow = np.array([23,50,100])
        upper_yellow = np.array([50,255,255])
        yellow_mask = cv2.inRange(self.frame_HSV, lower_yellow, upper_yellow)
        return yellow_mask
    
    def green_det(self):
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(self.frame_HSV, lower_green, upper_green)
        return green_mask
    
    def black_det(self):
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])  # V=50 is a good start for black, adjust if needed
        black_mask = cv2.inRange(self.frame_HSV, lower_black, upper_black)
        return black_mask
        
    def getOrientation(self, pts, img):
        ## [pca]
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        
        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
        ## [pca]
        
        ## [visualization]
        
        angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        ## [visualization]
        
        # Label with the rotation angle
        ang = -int(np.rad2deg(angle)) - 90
        if ang> -210 and ang < -150:
            pass
        else:
            # Draw the principal components
            cv2.circle(img, cntr, 3, (255, 0, 255), 2)
            p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
            p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        return angle
    
    def is_contour_on_side(self, contour, side="left"):
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        if side == "left":
            return STEER_LEFT
        elif side == "right":
            return STEER_RIGHT
        return False
    
    def steer(self,pulse):
        self.pi.set_servo_pulsewidth(STEER_PIN, pulse)

    def drive(self, pulse):
        self.pi.set_servo_pulsewidth(DRIVE_PIN, pulse)

    def arrow_det(self):   
        edged1 = cv2.Canny(self.black_mask, threshold1, threshold2)
        kernel = np.ones((3, 3))
        img_dilate = cv2.dilate(edged1, kernel, iterations=2)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)
        # Convert image to binary
        _, bw = cv2.threshold(img_erode, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Find all the contours in the thresholded image
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i, c in enumerate(contours):
        # Calculate the area of each contour
            area = cv2.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 7000:
                continue
    
            var = self.getOrientation(c, self.frame)
            if var > 0:
                return STEER_LEFT
            else:
                return STEER_RIGHT
        return STEER_CENTER  # If no arrow found

    def main(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.running = True
        self.pi = pigpio.pi()
        ret, self.frame = self.cap.read()
        if not ret:
            print("Can't receive initial frame. Exiting ...")
            return
        self.height, self.width = self.frame.shape[:2]
        if not self.pi.connected:
            print("Pi is not running")
            exit()
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.steer(STEER_CENTER)
        self.last_steer = STEER_CENTER
        self.last_drive = DRIVE_FORWARD
        try:
            while True:
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break 
                self.frame_HSV = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV) # All columns

                # Store for later use
                blue_mask = self.blue_det()
                yellow_mask = self.yellow_det()
                green_mask = self.green_det()
                self.black_mask = self.black_det()

                # Just before contour detection:
                roi_height = self.height // 4  # Use the bottom third
                blue_mask_roi = blue_mask[-roi_height:, :]
                yellow_mask_roi = yellow_mask[-roi_height:, :]
                green_mask_roi =  green_mask[-roi_height:, :]

                # Find contours for blue and yellow masks
                contours_blue, _ = cv2.findContours(blue_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_yellow, _ = cv2.findContours(yellow_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_green, _ = cv2.findContours(green_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

                blue_x = None
                yellow_x = None
                green = None

                # Find the largest blue contour (right line)
                # Check for side-consistent blue and yellow
                blue_valid = False
                yellow_valid = False

                if contours_blue:
                    largest_blue = max(contours_blue, key=cv2.contourArea)
                    if cv2.contourArea(largest_blue) > MIN_CONTOUR_AREA:
                        if self.is_contour_on_side(largest_blue, "right"):
                            M_blue = cv2.moments(largest_blue)
                            if M_blue["m00"] != 0:
                                blue_x = int(M_blue["m10"] / M_blue["m00"])
                                blue_valid = True

                if contours_yellow:
                    largest_yellow = max(contours_yellow, key=cv2.contourArea)
                    if cv2.contourArea(largest_yellow) > MIN_CONTOUR_AREA:
                        if self.is_contour_on_side(largest_yellow, "left"):
                            M_yellow = cv2.moments(largest_yellow)
                            if M_yellow["m00"] != 0:
                                yellow_x = int(M_yellow["m10"] / M_yellow["m00"])
                                yellow_valid = True

                if contours_green:
                    largest_green = max(contours_green, key=cv2.contourArea)
                    M_green = cv2.moments(largest_green)
                    if M_green["m00"] != 0:
                        green_x = int(M_green["m10"] / M_green["m00"])
                green_x = None
                try:
                    # Decide steering
                    if green_x is not None and green_x < self.width * 0.6 and green_x > self.width * 0.3:
                        self.drive(DRIVE_STOP)
                        print("Green Detected, stopping")
                        break
                    elif np.count_nonzero(self.black_mask) > 500:
                        print("Arrow Detected")
                        angle = self.arrow_det()
                    elif blue_valid and yellow_valid:
                        print("Straight")
                        center = (blue_x + yellow_x) // 2
                        frame_center = self.width // 2
                        offset = center - frame_center
                        if abs(offset) < 20:
                            self.steer(STEER_CENTER)
                            self.drive(DRIVE_FORWARD)
                            #time.sleep(0.5)
                            self.last_steer = STEER_CENTER
                            self.last_drive = DRIVE_FORWARD
                            #self.drive(DRIVE_STOP)
                            #time.sleep(0.4)
                        elif offset > 0:
                            self.steer(STEER_RIGHT)
                            self.drive(DRIVE_CORNER)
                            #time.sleep(0.3)
                            self.last_steer = STEER_RIGHT
                            self.last_drive = DRIVE_CORNER
                            #self.drive(DRIVE_STOP)
                            #time.sleep(0.3)
                        else:
                            self.steer(STEER_LEFT)
                            self.drive(DRIVE_CORNER)
                            #time.sleep(0.3)
                            self.last_steer = STEER_LEFT
                            self.last_drive = DRIVE_CORNER
                            #self.drive(DRIVE_STOP)
                            #time.sleep(0.3)
                    elif blue_valid:
                        print("Blue")
                        self.steer(STEER_LEFT)
                        self.drive(DRIVE_CORNER)
                        #time.sleep(0.3)
                        self.last_steer = STEER_LEFT
                        self.last_drive = DRIVE_CORNER
                        #self.drive(DRIVE_STOP)
                        #time.sleep(0.3)
                    elif yellow_valid:
                        print("Yellow")
                        self.steer(STEER_RIGHT)
                        self.drive(DRIVE_CORNER)
                        #time.sleep(0.3)
                        self.last_steer = STEER_RIGHT
                        self.last_drive = DRIVE_CORNER
                        #self.drive(DRIVE_STOP)
                        #time.sleep(0.3)
                    else:
                        print("None")
                        # No lines seen, continue last command
                        self.steer(self.last_steer)
                        self.drive(self.last_drive)
                        #time.sleep(0.3)
                        #self.drive(DRIVE_STOP)
                        #time.sleep(0.3)
                except Exception as e:
                    print(f"There was an error: {e}")
                    self.drive(DRIVE_STOP)
                    break
        except KeyboardInterrupt:
            print("Stopped by user (Ctrl+C)")
            self.drive(DRIVE_STOP)
            self.steer(STEER_CENTER)
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
                print(e)

Ben = Vision()
Ben.main()

#Call main only after input of x, run camera first then run movement afterwardss