"""
Ideas for movement
*Compare midpoints to the centre of the screen
Turn robot accordingly
If more than certain angle assume corner and slow down
Otherwise go full speed.
https://sourceforge.net/p/raspberry-gpio-python/wiki/Examples/ 
https://pypi.org/project/RPi.GPIO/
"""
import time
import pigpio

STEER_PIN = 12  # Change to your servo's GPIO pin

# Connect to pigpio daemon
pi = pigpio.pi()
if not pi.connected:
    print("pigpio daemon not running. Start with 'sudo pigpiod'")
    exit()

try:
    print("Sweeping servo from left to right...")
    for pulse in range(1000, 2001, 50):  # 1000 to 2000 in steps of 50
        pi.set_servo_pulsewidth(STEER_PIN, pulse)
        print(f"Pulse: {pulse}")
        time.sleep(0.2)

    print("Sweeping back from right to left...")
    for pulse in range(2000, 999, -50):
        pi.set_servo_pulsewidth(STEER_PIN, pulse)
        print(f"Pulse: {pulse}")
        time.sleep(0.2)

    print("Returning to center (1500)")
    pi.set_servo_pulsewidth(STEER_PIN, 1500)
    time.sleep(1)

finally:
    print("Turning off servo signal.")
    pi.set_servo_pulsewidth(STEER_PIN, 0)
    pi.stop()