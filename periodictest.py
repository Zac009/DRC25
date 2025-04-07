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
contoursr, _ = cv2.findContours(edged2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursy, _ = cv2.findContours(edged1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ls = []
val = 20
split_lst = int(len(contoursr)/2)
for i in range(0, split_lst):
    ls.append(contoursr[i])
split_lst = int(len(contoursy)/2)
for i in range(0, split_lst):
    ls.append(contoursy[i])

contours = ls
ls = []
for x in contours:
    counter = len(x)//val
    for i in range(0,val):
        it = counter*i
        ls.append(x[it]) 
# Draw contours

drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (255,0,0), 2)

midpoints = []
for pt in ls:
    coor1 = pt[0][0]
    coor2 = pt[0][1]
    cv2.circle(drawing, (coor1, coor2), 4, (0,255,0), -1)

# Compute midpoints
# Draw midpoints
for pt in midpoints:
    cv2.circle(drawing, pt, 4, (255, 255, 255), -1)  # White for midpoints

cv2.imshow("Line Detection",drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()