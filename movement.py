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

# GPIO pin setup
STEER_PIN = 12
DRIVE_PIN = 16

# Servo pulse values
STEER_LEFT = 1000
STEER_CENTER = 1500
STEER_RIGHT = 2000

DRIVE_STOP = 1500
DRIVE_FORWARD = 1600
DRIVE_BACKWARD = 1400

# Connect to pigpio daemon
pi = pigpio.pi()
if not pi.connected:
    print("Pi is not running")
    exit()

# Helper functions
def steer(pulse):
    pi.set_servo_pulsewidth(STEER_PIN, pulse)

def drive(pulse):
    pi.set_servo_pulsewidth(DRIVE_PIN, pulse)

# Autonomous logic starts here
try:
    print("Starting autonomous routine...")
    
    # Move forward
    steer(STEER_CENTER)
    drive(DRIVE_FORWARD)
    time.sleep(1)

    # Turn left while moving
    steer(STEER_RIGHT)
    time.sleep(1)

    # Stop
    drive(DRIVE_STOP)
    steer(STEER_CENTER)

    print("Autonomous routine complete.")

finally:
    # Cleanup
    drive(DRIVE_STOP)
    steer(STEER_CENTER)
    pi.set_servo_pulsewidth(DRIVE_PIN, 0)
    pi.set_servo_pulsewidth(STEER_PIN, 0)
    pi.stop()