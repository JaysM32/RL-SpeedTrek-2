#!/usr/bin/env python3
"""Keyboard control for the JetRacer, over SSH or a JupyterLab terminal.

  w / s      throttle up / down (one step per press)
  a / d      steer left / right (one step per press)
  c          centre steering
  space      STOP (throttle 0)
  [ / ]      adjust steering trim (offset) if the car drifts when centred
  - / =      decrease / increase steering gain
  q          quit (always stops the car)
"""
import curses
import time

from jetracer.nvidia_racecar import NvidiaRacecar

MAX_THROTTLE = 0.3   # hard cap, raise only once you're comfortable
THROTTLE_STEP = 0.02
STEER_STEP = 0.1


def main(screen):
    car = NvidiaRacecar()
    car.throttle = 0.0
    car.steering = 0.0
    throttle, steering = 0.0, 0.0

    screen.nodelay(True)
    curses.curs_set(0)
    try:
        while True:
            key = screen.getch()
            if key == ord('q'):
                break
            elif key == ord(' '):
                throttle = 0.0
            elif key == ord('w'):
                throttle = min(throttle + THROTTLE_STEP, MAX_THROTTLE)
            elif key == ord('s'):
                throttle = max(throttle - THROTTLE_STEP, -MAX_THROTTLE)
            elif key == ord('a'):
                steering = max(steering - STEER_STEP, -1.0)
            elif key == ord('d'):
                steering = min(steering + STEER_STEP, 1.0)
            elif key == ord('c'):
                steering = 0.0
            elif key == ord('['):
                car.steering_offset -= 0.01
            elif key == ord(']'):
                car.steering_offset += 0.01
            elif key == ord('-'):
                car.steering_gain *= 0.95
            elif key == ord('='):
                car.steering_gain *= 1.05

            car.throttle = throttle
            car.steering = steering

            screen.erase()
            screen.addstr(0, 0, __doc__)
            screen.addstr(13, 0, "throttle: %+.2f   (cap %.2f)" % (throttle, MAX_THROTTLE))
            screen.addstr(14, 0, "steering: %+.2f" % steering)
            screen.addstr(15, 0, "steering_gain: %+.3f   steering_offset: %+.3f"
                          % (car.steering_gain, car.steering_offset))
            screen.addstr(16, 0, "throttle_gain: %+.3f" % car.throttle_gain)
            screen.refresh()
            time.sleep(0.05)
    finally:
        car.throttle = 0.0
        car.steering = 0.0


if __name__ == "__main__":
    curses.wrapper(main)