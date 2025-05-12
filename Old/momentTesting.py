import cv2
import numpy as np
import math
import random as rng
import argparse

#cv2 MAT function
"""
Get the values of all pixels (Black or white [1,0])
Work out the distance based on pixels(camera specs will define the true distance)
Find midpoint 
Algorithm for corners:
    -If the np array has certain 1 values (On a diagnol), you can assume corner
    -Work out motor functions based on the angle of the diagnol
"""

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

# Read Image
image = cv2.imread(r'track.png')
# Resize width=500 height=300 incase of inputting raspi captured image
image = cv2.resize(image,(r_width,r_height))
frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100,50,120])
upper_blue = np.array([150,255,255])
blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)
lower_yellow = np.array([25,30,100])
upper_yellow = np.array([40,255,255])
yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
mask = cv2.add(yellow_mask, blue_mask)

edged1 = cv2.Canny(yellow_mask, threshold1, threshold2)
edged2 = cv2.Canny(blue_mask, threshold1, threshold2)
canny_output = cv2.Canny(mask, threshold1, threshold2)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get the moments
mu = [None]*len(contours)
for i in range(len(contours)):
    mu[i] = cv2.moments(contours[i])

# Get the mass centers
mc = [None]*len(contours)
for i in range(len(contours)):
    # add 1e-5 to avoid division by zero
    mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    print("BLAH")
print(mu)

# Draw contours

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2)
    cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)

center_1 = (int(mc[0][0]), int(mc[0][1]))
center_2 = (int(mc[1][0]), int(mc[1][1]))
# Calculate the area with the moments 00 and compare with the result of the OpenCV function
for i in range(len(contours)):
    print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)))
 
midpoint_x = (center_1[0] + center_2[0]) // 2
midpoint_y = (center_1[1] + center_2[1]) // 2
cv2.circle(drawing, (midpoint_x, midpoint_y), 10, (255, 255, 0), -1)

cv2.imshow("Line Detection",drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()