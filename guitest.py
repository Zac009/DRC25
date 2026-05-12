import cv2
import numpy as np

class HSVAdjuster:
    def __init__(self):
        self.window_name = "HSV Threshold Adjuster"
        self.color_presets = {
            'b': ('Blue', np.array([100, 50, 120]), np.array([150, 255, 255])),
            'y': ('Yellow', np.array([30, 50, 100]), np.array([50, 255, 255])),
            'g': ('Green', np.array([35, 100, 100]), np.array([85, 255, 255])),
            'p': ('Purple', np.array([120, 70, 70]), np.array([160, 255, 255]))
        }
        self.lower = self.color_presets['b'][1].copy()
        self.upper = self.color_presets['b'][2].copy()

        # Create window and trackbars
        cv2.namedWindow(self.window_name)
        self.create_trackbars()

    def create_trackbars(self):
        cv2.createTrackbar("Lower H", self.window_name, self.lower[0], 255, lambda x: None)
        cv2.createTrackbar("Lower S", self.window_name, self.lower[1], 255, lambda x: None)
        cv2.createTrackbar("Lower V", self.window_name, self.lower[2], 255, lambda x: None)

        cv2.createTrackbar("Upper H", self.window_name, self.upper[0], 255, lambda x: None)
        cv2.createTrackbar("Upper S", self.window_name, self.upper[1], 255, lambda x: None)
        cv2.createTrackbar("Upper V", self.window_name, self.upper[2], 255, lambda x: None)

    def update_trackbars(self, lower, upper):
        cv2.setTrackbarPos("Lower H", self.window_name, lower[0])
        cv2.setTrackbarPos("Lower S", self.window_name, lower[1])
        cv2.setTrackbarPos("Lower V", self.window_name, lower[2])
        cv2.setTrackbarPos("Upper H", self.window_name, upper[0])
        cv2.setTrackbarPos("Upper S", self.window_name, upper[1])
        cv2.setTrackbarPos("Upper V", self.window_name, upper[2])

    def get_thresholds(self):
        lh = cv2.getTrackbarPos("Lower H", self.window_name)
        ls = cv2.getTrackbarPos("Lower S", self.window_name)
        lv = cv2.getTrackbarPos("Lower V", self.window_name)
        uh = cv2.getTrackbarPos("Upper H", self.window_name)
        us = cv2.getTrackbarPos("Upper S", self.window_name)
        uv = cv2.getTrackbarPos("Upper V", self.window_name)
        return np.array([lh, ls, lv]), np.array([uh, us, uv])

    def run(self):
        cap = cv2.VideoCapture(1)  # Use 0 or 1 depending on your camera
        print("Press 'b' (Blue), 'y' (Yellow), 'g' (Green), 'p' (Purple) to switch colors.")
        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.lower, self.upper = self.get_thresholds()

            mask = cv2.inRange(frame_HSV, self.lower, self.upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow(self.window_name, result)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif chr(key) in self.color_presets:
                name, lower, upper = self.color_presets[chr(key)]
                print(f"Switched to {name}")
                self.update_trackbars(lower, upper)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HSVAdjuster()
    app.run()
