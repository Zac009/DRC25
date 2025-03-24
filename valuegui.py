import tkinter as tk
import numpy as np
import cv2
import threading

class Value:
    def __init__(self):
        self.lower_blue = np.array([100,50,120])
        self.upper_blue = np.array([150,255,255])
        self.lower_yellow = np.array([25,30,100])
        self.upper_yellow = np.array([40,255,255])

# Function to update the color detection based on slider values
    def update_thresholds(self):
        global lower_hue, upper_hue, lower_saturation, upper_saturation, lower_value, upper_value

        # Get current slider values
        self.lower_blue = np.array([self.lower_blue_slider_red.get(),self.lower_blue_slider_green.get(),self.lower_blue_slider_blue.get()])
        self.upper_blue = np.array([self.upper_blue_slider_red.get(),self.upper_blue_slider_green.get(),self.upper_blue_slider_blue.get()])
        return self.lower_blue, self.upper_blue

    def main(self):
        window = tk.Tk()
        window.geometry("400x400")
        window.title("HSV Color Detection")


        # Hue sliders
        hue_label = tk.Label(window, text="Color Detection Range")
        hue_label.pack()

        self.lower_blue_slider_red = tk.Scale(window, from_=0, to=255, orient="horizontal", label="Lower Blue Red")
        self.lower_blue_slider_red.pack()
        self.lower_blue_slider_red.set(100)

        self.lower_blue_slider_green = tk.Scale(window, from_=0, to=255, orient="horizontal", label="Lower Blue Green")
        self.lower_blue_slider_green.pack()
        self.lower_blue_slider_green.set(50)

        self.lower_blue_slider_blue = tk.Scale(window, from_=0, to=255, orient="horizontal", label="Lower Blue Blue")
        self.lower_blue_slider_blue.pack()
        self.lower_blue_slider_blue.set(120)

        self.upper_blue_slider_red = tk.Scale(window, from_=0, to=255, orient="horizontal", label="Upper Blue Red")
        self.upper_blue_slider_red.pack()
        self.upper_blue_slider_red.set(150)

        self.upper_blue_slider_blue = tk.Scale(window, from_=0, to=255, orient="horizontal", label="Upper Blue green")
        self.upper_blue_slider_blue.pack()
        self.upper_blue_slider_blue.set(255)

        self.upper_blue_slider_green = tk.Scale(window, from_=0, to=255, orient="horizontal", label="Upper Blue Blue")
        self.upper_blue_slider_green.pack()
        self.upper_blue_slider_green.set(255)


        # Apply button to update parameters
        apply_button = tk.Button(window, text="Apply", command=self.update_thresholds)
        apply_button.pack()

        window.mainloop()

test = Value()
test.main()