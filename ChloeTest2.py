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

MIN_CONTOUR_AREA = 200  # Tune this value

class Vision:
    def __init__(self):
        self.r_width = 500
        self.r_height = 300
        self.direction = "Blue"

    def blue_det(self):
        lower_blue = np.array([90,50,120])
        upper_blue = np.array([150,255,255])
        blue_mask = cv2.inRange(self.frame_HSV, lower_blue, upper_blue)
        return blue_mask
    
    def yellow_det(self):
        lower_yellow = np.array([15,50,100])
        upper_yellow = np.array([50,255,255])
        yellow_mask = cv2.inRange(self.frame_HSV, lower_yellow, upper_yellow)
        return yellow_mask
    
    def green_det(self):
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(self.frame_HSV, lower_green, upper_green)
        return green_mask
    
    def steer(self,pulse):
        self.pi.set_servo_pulsewidth(STEER_PIN, pulse)

    def drive(self, pulse):
        self.pi.set_servo_pulsewidth(DRIVE_PIN, pulse)

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
                green_x = None
                green = None

                # Find the largest blue contour (right line)
                if contours_blue:
                    largest_blue = max(contours_blue, key=cv2.contourArea)
                    if cv2.contourArea(largest_blue) > MIN_CONTOUR_AREA:
                        M_blue = cv2.moments(largest_blue)
                        if M_blue["m00"] != 0:
                            blue_x = int(M_blue["m10"] / M_blue["m00"])

                # Find the largest yellow contour (left line)
                if contours_yellow:
                    largest_yellow = max(contours_yellow, key=cv2.contourArea)
                    M_yellow = cv2.moments(largest_yellow)
                    if M_yellow["m00"] != 0:
                        yellow_x = int(M_yellow["m10"] / M_yellow["m00"])

                if contours_green:
                    largest_green = max(contours_green, key=cv2.contourArea)
                    M_green = cv2.moments(largest_green)
                    if M_green["m00"] != 0:
                        green_x = int(M_green["m10"] / M_green["m00"])

                try:
                    # Decide steering
                    if green_x is not None and green_x < self.width * 0.6 and green_x > self.width * 0.3:
                        self.drive(DRIVE_STOP)
                        break
                    elif blue_x is not None and yellow_x is not None:
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
                    elif blue_x is not None:
                        print("Blue")
                        self.steer(STEER_LEFT)
                        self.drive(DRIVE_CORNER)
                        #time.sleep(0.3)
                        self.last_steer = STEER_LEFT
                        self.last_drive = DRIVE_CORNER
                        #self.drive(DRIVE_STOP)
                        #time.sleep(0.3)
                    elif yellow_x is not None:
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