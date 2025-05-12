import cv2
import numpy as np

def mask1(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def mask2(frame):
    return cv2.Canny(frame, 100, 200)

def mask3(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def mask4(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    return cv2.inRange(hsv, lower_red, upper_red)