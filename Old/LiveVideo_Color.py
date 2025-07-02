import numpy as np
import cv2 as cv2

cap = cv2.VideoCapture(0)

# Parameters
minLineLength = 25
maxLineGap = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output11.avi', fourcc, 20.0, (640, 480))

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # HSV color filtering
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 120])
    upper_blue = np.array([150, 255, 255])
    lower_yellow = np.array([22, 50, 100])
    upper_yellow = np.array([50, 255, 255])

    blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)
    yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
    mask = cv2.add(yellow_mask, blue_mask)


    # Display result
    cv2.imshow('lines_and_curves', mask)
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
