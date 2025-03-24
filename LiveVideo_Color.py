import numpy as np
import cv2 as cv2
import threading


cap = cv2.VideoCapture(0)
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
    height, width, _ = frame.shape
    right_image = frame[:, width//2:]
    frame_HSV = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,50,120])
    upper_blue = np.array([150,255,255])
    blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)
    lower_yellow = np.array([25,30,100])
    upper_yellow = np.array([40,255,255])
    yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
    mask = cv2.add(yellow_mask, blue_mask)

    edged1 = cv2.Canny(yellow_mask, threshold1, threshold2)
    edged2 = cv2.Canny(blue_mask, threshold1, threshold2)
    out.write(mask)
 
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', mask)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()