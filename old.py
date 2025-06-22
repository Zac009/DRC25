def main(self):
    self.running = True
    cap = cv2.VideoCapture(0)

    while self.running:
        ret, frame = cap.read()
        if not ret:
            break

        midpoint = self.detect_midpoint(frame)

        if midpoint is not None:
            offset = midpoint[0] - FRAME_CENTER_X
            k = 0.005
            angle = -offset * k
            self.send_drive_command(speed=0.2, angle=angle)
        else:
            self.send_drive_command(speed=0, angle=0)  # stop or slow drift

        time.sleep(0.05)  # Short delay to avoid spamming commands

    cap.release()
    self.send_drive_command(speed=0, angle=0)  # Full stop at end
