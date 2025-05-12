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

edged1 = cv2.Canny(yellow_mask, threshold1, threshold2)
edged2 = cv2.Canny(blue_mask, threshold1, threshold2)
# Detect points that form a line
yellow_lines = cv2.HoughLinesP(edged1,1,np.pi/180,max_slider,minLineLength,maxLineGap)
blue_lines = cv2.HoughLinesP(edged2,1,np.pi/180,max_slider,minLineLength,maxLineGap)
# Accumulate the weighted sum of angles and line lengths
total_theta = 0
total_length = 0

# Accumulate angles and lengths for yellow lines
for x in range(0, len(yellow_lines)):
    for x1, y1, x2, y2 in yellow_lines[x]:
        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),3)
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += line_length
        total_theta += math.atan2((y2 - y1), (x2 - x1)) * line_length

# Accumulate angles and lengths for blue lines
for x in range(0, len(blue_lines)):
    for x1, y1, x2, y2 in blue_lines[x]:
        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),3)
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += line_length
        total_theta += math.atan2((y2 - y1), (x2 - x1)) * line_length
print(total_theta)

# Calculate the weighted average angle
if total_length > 0:
    average_theta = total_theta / total_length
else:
    average_theta = 0

# Determine direction based on the average angle
threshold=5
#S = |226| < t
#L = 222 > t
#R = 523 > t
if(theta>threshold):
    print("Go left")
if(theta<-threshold):
    print("Go right")
if(abs(theta)<threshold):
    print("Go straight")
theta=0
cv2.imshow("Line Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()