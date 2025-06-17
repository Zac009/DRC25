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
import curses

STEER_PIN = 12
DRIVE_PIN = 16

STEER_LEFT = 1000
STEER_CENTER = 1500
STEER_RIGHT = 2000

DRIVE_STOP = 1500
DRIVE_FORWARD = 1600
DRIVE_BACKWARD = 1400

pi = pigpio.pi()
if not pi.connected:
    print("Pi motors not connected.")
    exit()

def steer(pulse):
    pi.set_servo_pulsewidth(STEER_PIN, pulse)

def drive(pulse):
    pi.set_servo_pulsewidth(DRIVE_PIN, pulse)

def main(stdscr):
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.keypad(True)
    stdscr.clear()
    stdscr.addstr(0, 0, "Use W/A/S/D to control. Press Q to quit.")

    steer(STEER_CENTER)
    drive(DRIVE_STOP)

    try:
        while True:
            key = stdscr.getch()
            if key == ord('w'):
                drive(DRIVE_FORWARD)
            elif key == ord('s'):
                drive(DRIVE_BACKWARD)
            else:
                drive(DRIVE_STOP)

            if key == ord('a'):
                steer(STEER_LEFT)
            elif key == ord('d'):
                steer(STEER_RIGHT)
            else:
                steer(STEER_CENTER)

            if key == ord('q'):
                break

            time.sleep(0.05)

    finally:
        drive(DRIVE_STOP)
        steer(STEER_CENTER)
        pi.set_servo_pulsewidth(DRIVE_PIN, 0)
        pi.set_servo_pulsewidth(STEER_PIN, 0)
        pi.stop()

curses.wrapper(main)