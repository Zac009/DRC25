import cv2
import numpy as np
import math

#cv2 MAT function
"""
Get the values of all pixels (Black or white [1,0]) - Done
Work out the distance based on pixels(camera specs will define the true distance) - Pending
Find midpoint - Done
Algorithm for corners:
    -If the np array has certain 1 values (On a diagnol), you can assume corner
    -Work out motor functions based on the angle of the diagnol
Method 2 - 
    - Invert the matrix (np.flip(xxx))
        - Same method will get column distances instead of rows
    - If 2 values are in the column assume corner has been taken
    - Problem for slants though
        - Potential have horizontal pairs have a heigher priority over the vertical pairs. This should mean that slants are taken at "striaght" rather then a turn
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
# Detect points that form a line
dis = []
disl = []
disr = []
count = 0
res = 0
used = False
for i in mask:
    for x in i:
        if x == 255 and i[count-1] != 255:
            dis.append(count)
            res += 1
            used = True
            if res > 2:
                dis.pop(-2)
        count += 1
    if not used:
        dis.append(0)
    used = False
    res = 0
    count = 0

even = True
for i in dis:
    if even:
        disl.append(i)
        even = False
    else:
        disr.append(i)
        even = True

final = []
for i in range(len(disl)):
    result = disr[i] + disl[i]
    result /= 2
    final.append(result)

count = 1
for i in final:
    cv2.circle(image, (int(i),count), radius=0, color=(0, 0, 0), thickness=-2)
    count += 1

print(final)
cv2.imshow("Line Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()