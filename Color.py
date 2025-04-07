import cv2
import numpy as np
import math

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
image = cv2.imread(r'test2.png')
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
# Detect points that form a line

cv2.imshow("Line Detection",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()