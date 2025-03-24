import cv2

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Read frames in a loop
while(True):
    # Read a frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Load stereo image (assuming side-by-side format)
    # Split into left and right images
    height, width, _ = frame.shape
    left_image = frame[:, :width//2]
    right_image = frame[:, width//2:]

    # Convert both images to grayscale
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Compute the average image
    mono_image = cv2.addWeighted(left_gray, 0.50, right_gray, 0.5, 0)

    # Display the frame
    cv2.imshow('frame', right_image)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()