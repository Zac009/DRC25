import threading
from Old.valuegui import Value
import cv2
import numpy as np
import math
gui = Value()
def start_gui():
    gui.main()

gui_thread = threading.Thread(target=start_gui)
gui_thread.start()

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
def process_image():
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

    while True:
        cv2.imshow("Line Detection", mask)

        # Wait for a key press to break the loop and exit (this is necessary to avoid blocking the GUI)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Run the image processing function in a separate thread
image_thread = threading.Thread(target=process_image)
image_thread.start()

# Optionally, you could keep the program running in the main thread, 
# allowing both the GUI and image processing to execute concurrently.
# The main thread will continue executing this after starting the image thread.
image_thread.join()
gui_thread.join()  # Wait for the image processing thread to finish