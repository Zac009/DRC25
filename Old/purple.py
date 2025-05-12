import numpy as np
import cv2 as cv2
import threading
import numpy as np


cap = cv2.VideoCapture('qut_demo.mov')

threshold1 = 85
threshold2 = 85
theta=0
r_width = 500
r_height = 300
minLineLength = 10 #10
maxLineGap = 1 #1
k_width = 5
k_height = 5
max_slider = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (640,  480))
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
    # Resize width=500 height=300 incase of inputting raspi captured image
    """height, width, _ = frame.shape
    right_image = frame[:, width//2:]"""
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 70, 70])   # H, S, V
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(frame_HSV, lower_purple, upper_purple)
    contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = cv2.cvtColor(np.zeros_like(purple_mask), cv2.COLOR_GRAY2BGR)
    # Draw contours on the original frame
    cv2.drawContours(final, contours, -1, (255, 255, 255), 2) 
    # Our operations on the frame come here
    # Display the resulting frame
    """
    Check if
    """
    out.write(purple_mask)
    cv2.imshow('frame', final)
    cv2.imshow('scan', purple_mask)
    cv2.moveWindow("scan", 0, 500)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()