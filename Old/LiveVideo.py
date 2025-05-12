import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture(1)
 
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
ret, frame = cap.read()
height, width = frame.shape[:2]
out = cv.VideoWriter('output20.avi', fourcc, 20.0, (width, height))
 
while cap.isOpened():
    if not out.isOpened():
        print("VideoWriter failed to open")
        exit()
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 0)
 
    # Convert to grayscale and back to BGR
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    out.write(gray_bgr)
 
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
out.release()
cv.destroyAllWindows()