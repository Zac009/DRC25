import numpy as np
import cv2 as cv2

cap = cv2.VideoCapture("qut_demo.mov")

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
    lower_yellow = np.array([30, 50, 100])
    upper_yellow = np.array([50, 255, 255])

    blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)
    yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
    mask = cv2.add(yellow_mask, blue_mask)
    new_mask = np.zeros_like(mask)

    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- LINE DETECTION ---
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(new_mask, (x1, y1), (x2, y2), 255, 2)  # green lines

    # --- CURVE / CONTOUR DETECTION ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:  # adjust threshold as needed
            cv2.drawContours(new_mask, [cnt], -1, 255, 2)

    # Display result
    cv2.imshow('lines_and_curves', new_mask)
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
