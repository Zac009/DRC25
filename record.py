import cv2

# Setup
cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 on macOS
out = cv2.VideoWriter('output11.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)  # Save frame to file
    cv2.imshow('Recording...', frame)

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to stop
        break

cap.release()
out.release()
cv2.destroyAllWindows()