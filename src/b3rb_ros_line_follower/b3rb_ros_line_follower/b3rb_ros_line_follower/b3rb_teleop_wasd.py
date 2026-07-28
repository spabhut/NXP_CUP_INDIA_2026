#!/usr/bin/env python3
# Copyright 2024-2026 NXP

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import sys
import select
import termios
import tty

BANNER = """
---------------------------------------------------
      NXP B3RB Buggy WASD Joy-Teleop Node
---------------------------------------------------
Controls:
    W : Move Forward (Increase Throttle)
    S : Move Backward (Reverse)
    A : Turn Left (Steer Left)
    D : Turn Right (Steer Right)
    
 Space : Emergency Brake (Stop all motion)
    Q : Quit Teleop Node
---------------------------------------------------
"""

SPEED_MAX = 5  # Maximum throttle percentage/value
TURN_MAX = 3   # Maximum steering angle percentage/value

class B3RBJoyWASDTeleop(Node):
    def __init__(self):
        super().__init__('b3rb_teleop_wasd')

        # Publisher sending direct Joy messages to Cerebri
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            10
        )

        self.target_speed = 0.0
        self.target_turn = 0.0

        self.speed_step = 0.1
        self.turn_step = 0.2

        # Send Joy heartbeats continuously at 20 Hz
        self.timer = self.create_timer(0.05, self.publish_drive_commands)
        self.get_logger().info("B3RB Joy Teleop Node Active.")

    def publish_drive_commands(self):
        """Periodically publishes Joy commands with active manual arming buttons."""
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Active arming buttons bitmask to prevent joy loss failsafe in Cerebri
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, float(self.target_speed), 0.0, float(self.target_turn)]
        self.publisher_joy.publish(msg)

    def get_key(self, settings):
        """Captures raw non-blocking terminal keypresses."""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key.lower()

    def run_teleop_loop(self):
        """Main WASD control loop."""
        settings = termios.tcgetattr(sys.stdin)
        print(BANNER)
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
                key = self.get_key(settings)

                if key == 'w':
                    self.target_speed = min(SPEED_MAX, self.target_speed + self.speed_step)
                elif key == 's':
                    self.target_speed = max(-SPEED_MAX, self.target_speed - self.speed_step)
                elif key == 'a':
                    self.target_turn = min(TURN_MAX, self.target_turn + self.turn_step)
                elif key == 'd':
                    self.target_turn = max(-TURN_MAX, self.target_turn - self.turn_step)
                elif key == ' ':
                    self.target_speed = 0.0
                    self.target_turn = 0.0
                    print("\r[BRAKE] Emergency Brake Applied!       ", end="")
                elif key == 'q':
                    print("\rExiting Teleop...                       ")
                    break

                # Self-centering steering when A/D keys are released
                if key not in ['a', 'd']:
                    self.target_turn *= 0.5
                    if abs(self.target_turn) < 0.01:
                        self.target_turn = 0.0

                if key != '':
                    print(f"\rThrottle (Axis 1): {self.target_speed:.2f} | Steering (Axis 3): {self.target_turn:.2f}  ", end="")

        except Exception as e:
            self.get_logger().error(f"Teleop error: {e}")

        finally:
            self.target_speed = 0.0
            self.target_turn = 0.0
            self.publish_drive_commands()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    rclpy.init(args=args)
    node = B3RBJoyWASDTeleop()
    
    try:
        node.run_teleop_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()