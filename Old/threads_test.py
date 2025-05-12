import threading
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import queue

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Create a label to display the OpenCV frame
        self.label = tk.Label(window)
        self.label.pack()
        
        # Create a queue for frame data to pass between threads
        self.queue = queue.Queue()

        # Start OpenCV thread
        self.video_thread = threading.Thread(target=self.video_capture)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Start Tkinter update loop
        self.update_gui()

    def video_capture(self):
        # Capture video from the webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame (e.g., convert to grayscale)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add the processed frame to the queue
            self.queue.put(gray)

        cap.release()

    def update_gui(self):
        try:
            # If there is a new frame in the queue, display it
            if not self.queue.empty():
                frame = self.queue.get()

                # Convert frame to ImageTk format
                image = Image.fromarray(frame)
                image = ImageTk.PhotoImage(image)

                # Update the label with the new image
                self.label.config(image=image)
                self.label.image = image

        except Exception as e:
            print(f"Error updating GUI: {e}")

        # Call this function again after a delay (50ms)
        self.window.after(50, self.update_gui)

# Create the Tkinter window
root = tk.Tk()

# Create the application
app = Application(root, "OpenCV and Tkinter")

# Start the Tkinter main loop
root.mainloop()