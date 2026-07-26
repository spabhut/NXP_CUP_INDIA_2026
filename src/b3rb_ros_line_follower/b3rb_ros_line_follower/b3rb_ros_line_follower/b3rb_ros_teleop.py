#!/usr/bin/env python3
# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simple keyboard teleop node for the NXP CUP B3RB buggy.

IMPORTANT: the B3RB is a car-like (Ackermann-steered) buggy, not a
differential/skid-steer robot. The "turn" axis is a front-wheel STEERING
ANGLE, not a body rotation rate -- so turning in place with zero forward
speed does nothing, exactly like turning a parked car's steering wheel with
the engine off. That's why a/d need a little forward creep combined with
full steering lock to actually carve a turn.

Keys:
  w : drive straight forward
  s : drive straight backward
  a : creep forward + full steering lock left
  d : creep forward + full steering lock right
  space : stop
  q : quit teleop and hand control back to the autonomous node

Publishes to /cerebri/in/joy using the same Joy format line_follower's
manual-override mode expects. Also publishes std_msgs/Bool on
/teleop/override so line_follower.py knows to stop publishing its own drive
commands while this node is running -- otherwise the two nodes would both
write to /cerebri/in/joy and fight each other.
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

QOS_PROFILE_DEFAULT = 10

# Fixed commands -- tune these to taste.
DRIVE_SPEED = 1.0       # used for w (forward) / s (backward, negated)
TURN_CREEP_SPEED = 0.2  # small forward speed while turning (needed to steer at all)
TURN_ANGLE = 1.0        # full steering lock for a (left) / d (right, negated)

# If no key is pressed within this window, the buggy stops (safety).
KEY_TIMEOUT_SEC = 0.5

# How often we read a key / publish a command.
LOOP_HZ = 10.0
LOOP_PERIOD_SEC = 1.0 / LOOP_HZ

INSTRUCTIONS = """
NXP CUP B3RB Keyboard Teleop
----------------------------
  w : forward
  s : backward
  a : turn left  (creeps forward while steering)
  d : turn right (creeps forward while steering)
  space : stop
  q : quit (hands control back to the autonomous node)

Note: this buggy steers like a car (Ackermann) -- it can't rotate in place,
so a/d creep forward a little while steering, same as turning a real car.

While this is running, the autonomous line_follower node pauses its own
drive commands (it listens on /teleop/override) so the two nodes can't
fight over /cerebri/in/joy.
"""


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('b3rb_teleop')

        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

        self.publisher_override = self.create_publisher(
            Bool, '/teleop/override', QOS_PROFILE_DEFAULT)

        self.speed = 0.0
        self.turn = 0.0
        self.last_key_time = self.get_clock().now()
        self.running = True

        # Raw terminal mode so we can read single keypresses without Enter.
        self.settings = termios.tcgetattr(sys.stdin)

    def get_key(self):
        """Non-blocking single-character read from stdin (POSIX raw mode)."""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], LOOP_PERIOD_SEC)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def handle_key(self, key):
        if key == 'w':
            self.speed = DRIVE_SPEED
            self.turn = 0.0
        elif key == 's':
            self.speed = -DRIVE_SPEED
            self.turn = 0.0
        elif key == 'a':
            self.speed = TURN_CREEP_SPEED
            self.turn = TURN_ANGLE
        elif key == 'd':
            self.speed = TURN_CREEP_SPEED
            self.turn = -TURN_ANGLE
        elif key == ' ':
            self.speed = 0.0
            self.turn = 0.0
        elif key == 'q' or key == '\x03':  # 'q' or Ctrl-C
            self.running = False
            return
        else:
            return
        self.last_key_time = self.get_clock().now()

    def publish_drive_and_override(self):
        # Tell line_follower it must yield control while we drive.
        override_msg = Bool()
        override_msg.data = True
        self.publisher_override.publish(override_msg)

        # Safety auto-stop if nothing has been pressed recently.
        elapsed = (self.get_clock().now() - self.last_key_time).nanoseconds / 1e9
        if elapsed > KEY_TIMEOUT_SEC:
            self.speed = 0.0
            self.turn = 0.0

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, self.speed, 0.0, self.turn]
        self.publisher_joy.publish(msg)

    def release_control(self):
        """Zero the drive command and clear the override flag so the
        autonomous node safely resumes."""
        stop_msg = Joy()
        stop_msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        stop_msg.axes = [0.0, 0.0, 0.0, 0.0]

        release_msg = Bool()
        release_msg.data = False

        # Publish a few times since we're about to shut down and there's no
        # retry/ack mechanism for these two topics.
        for _ in range(5):
            self.publisher_joy.publish(stop_msg)
            self.publisher_override.publish(release_msg)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()

    print(INSTRUCTIONS)

    try:
        while node.running and rclpy.ok():
            key = node.get_key()
            if key:
                node.handle_key(key)
            node.publish_drive_and_override()
    except KeyboardInterrupt:
        pass
    finally:
        node.release_control()
        node.destroy_node()
        rclpy.shutdown()
        print("\nTeleop stopped. Control returned to the autonomous node.")


if __name__ == '__main__':
    main()