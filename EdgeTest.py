import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('qut_demo.mov')
def single_step_convolution(slice, filter):
    s = slice * filter
    cval = s.sum()
    return cval


def detect_edge(image, filter):

  (image_row, image_col) = np.shape(image)

  (filter_row, filter_col) =  np.shape(filter) # assuming the filter is square matrix

  padding = 0
  stride = 1

  Ht = image_row - filter_row +1
  Wd = image_col - filter_col +1

  detection_result = np.zeros((Ht, Wd))

  for H in range(Ht):
    for W in range(Wd):
      row_start = H
      row_end = H + filter_row
      col_start = W
      col_end = W + filter_col

      # Use the corners to define the image slice for convolution (≈1 line)
      slice = image[row_start:row_end, col_start:col_end]

      # Call the single_step_convolution() which you have implemented above
      # and pass image and filter as argument (≈1 lines)
      detection_result[H, W] = single_step_convolution(slice, filter)

      ### END CODE HERE ###

  return detection_result



try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the BGR image to HSV color space
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Display the original color image using cv2_imshow
        cv2.imshow("Original Color Image", frame)

        lower_yellow = np.array([23,50,100])
        upper_yellow = np.array([50,255,255])
        yellow_mask = cv2.inRange(frame_HSV, lower_yellow, upper_yellow)
        #cv2.imshow("Yellow Mask", yellow_mask)

        lower_blue = np.array([100,50,120])
        upper_blue = np.array([150,255,255])
        blue_mask = cv2.inRange(frame_HSV, lower_blue, upper_blue)
        #cv2.imshow("Blue Mask", blue_mask)

        final = cv2.bitwise_or(blue_mask, yellow_mask)
        #cv2.imshow("Combined Mask (Yellow and Blue)", final)


        Hfilter = np.asarray([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        Hedges = detect_edge(final, Hfilter)
        gX = cv2.convertScaleAbs(Hedges)

        Vfilter = np.asarray([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        Vedges = detect_edge(final, Vfilter)
        gY = cv2.convertScaleAbs(Vedges)


        gXY = cv2.convertScaleAbs(gX+gY)
        cv2.imshow("Combined Edges", gXY)
        # Wait for a key press to advance to the next frame. Press 'q' to quit.
        key = cv2.waitKey(1) # 0 means wait indefinitely for a key press
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
except KeyboardInterrupt:
        print("Interrupted by user")



