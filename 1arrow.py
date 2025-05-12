import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
final = []
threshold1 = 350
threshold2 = 350
theta=0
r_width = 500
r_height = 300
minLineLength = 5 #10
maxLineGap = 10 #1
k_width = 5
k_height = 5
max_slider = 10
def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  ang = -int(np.rad2deg(angle)) - 90
  if ang> -210 and ang < -150:
    pass
  else:
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
    final.append(ang)
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return angle
 
# Load the image
cap = cv.VideoCapture('ERERE.avi')
 
# Define the codec and create VideoWriter object
framerate = int(cap.get(cv.CAP_PROP_FPS))
framewidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameheight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'XVID')
filename = 'DRC2024.avi'
videoDimensions = (framewidth, frameheight)
recordedVideo = cv.VideoWriter(filename, fourcc, framerate, videoDimensions)
while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        # Our operations on the frame come here
    # Convert image to grayscale
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # given input image, kernel width =5 height = 5, Gaussian kernel standard deviation
    blurred = cv.GaussianBlur(gray, (k_width, k_height), 0)
    # Find the edges in the image using canny detector
    frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([0,0,0])
    upper_yellow = np.array([120,120,120])
    yellow_mask = cv.inRange(frame_HSV, lower_yellow, upper_yellow)

    edged1 = cv.Canny(yellow_mask, threshold1, threshold2)
    kernel = np.ones((3, 3))
    img_dilate = cv.dilate(edged1, kernel, iterations=2)
    img_erode = cv.erode(img_dilate, kernel, iterations=1)
    # Convert image to binary
    _, bw = cv.threshold(img_erode, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
    # Calculate the area of each contour
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        print(area)
        if area < 7000:
            continue
        
        # Draw each contour only for visualisation purposes
        cv.drawContours(img, contours, i, (0, 0, 255), 2)
        # Find the orientation of each shape
        getOrientation(c, img)
    cv.imshow("Line Detection",img)
    recordedVideo.write(img)
    
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
#out.release()
cv.destroyAllWindows()
  
# Save the output image to the current directory

if final[0] > 0:
  print("Left")
else: 
  print("Right")