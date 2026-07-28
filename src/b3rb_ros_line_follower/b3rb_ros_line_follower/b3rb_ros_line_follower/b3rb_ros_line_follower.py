# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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

import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String, Bool
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# Lane-following PID gains.
# error is normalized to roughly [-1, 1] (fraction of half-width), so gains
# stay in a similar scale to TURN_MIN/TURN_MAX. Tune KP first, then KD to
# damp oscillation; leave KI at 0 unless you see a persistent steady-state bias.
LANE_KP = 1.2
LANE_KI = 0.0
LANE_KD = 0.3
LANE_INTEGRAL_LIMIT = 1.0

# CONFIGURATION:
# The buggy is driven in manual mode by publishing standard controller Joy messages to /cerebri/in/joy.
# The layout is: msg.axes = [0.0, speed, 0.0, turn]
# - speed: positive for forward, negative for reverse. Range: [-1.0, 1.0]
# - turn: positive for left steer, negative for right steer. Range: [-1.0, 1.0]
# msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] (Keep buttons set to this pattern for manual override mode)

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    """

    # ------------------ Mission states ------------------
    STATE_EN_ROUTE = 'EN_ROUTE'
    STATE_AWAITING_ZONE = 'AWAITING_ZONE'
    STATE_IN_ZONE = 'IN_ZONE'
    STATE_SIGN_TURNING = 'SIGN_TURNING'

    # ------------------ LIDAR sectors (index range assuming a 360-sample scan; scaled otherwise) ------------------
    LEFT_SECTOR = (210, 330)    # building/zone check, left side
    RIGHT_SECTOR = (30, 150)    # building/zone check, right side

    # ------------------ Thresholds ------------------
    BUILDING_DIST_THRESHOLD = 2      # meters -- "close" for zone purposes
    BUILDING_OCCUPANCY_RATIO = 0.75     # fraction of a side sector that must be "close" to call it a building

    DEFAULT_SPEED = 0.5

    # ------------------ Destination name -> sign-board letter (per FAQ7) ------------------
    DESTINATION_TO_SIGN = {
        'PATIENT_1': 'A',
        'PATIENT_2': 'B',
        'PATIENT_3': 'C',
        'HOSPITAL_1': 'X',
        'HOSPITAL_2': 'Y',
        'HOSPITAL_3': 'Z',
    }

    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # Teleop override: when True, a human is driving via keyboard teleop, so this
        # node must NOT publish its own drive commands.
        self.subscription_teleop_override = self.create_subscription(
            Bool,
            '/teleop/override',
            self.teleop_override_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        self.publisher_target_destination = self.create_publisher(
            String,
            '/target_destination',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------
        self.target_speed = self.DEFAULT_SPEED
        self.target_turn = 0.0
        self.teleop_active = False

        # Lane-following PID state (used by edge_vectors_callback).
        self._lane_integral = 0.0
        self._lane_prev_error = 0.0
        self._last_turn = 0.0

        # Mission state machine
        self.state = self.STATE_EN_ROUTE

        # Mission target / QR tracking
        self.current_destination = "PATIENT_1"   # set by server instructions, e.g. "PATIENT_1"
        self.pending_qr_loc = None
        self.mission_completed = False

        # Server comms tracking
        self.own_uid = 0              # rolling uid (0-255) for messages *we* originate
        self.awaiting_ack_uid = None  # uid of our own outgoing message we're still waiting to be ack'd

        # Sign-board turn tracking: latched direction for the current STATE_SIGN_TURNING
        # episode (reset / actual turn maneuver TBD later).
        self.latched_sign_direction = None

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)
        self.publish_target_destination(self.current_destination)

        self.get_logger().info("Line Follower controller initialized. Safe Drive-Straight Mode active.")

    # ------------------ Drive helpers ------------------
    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        if self.teleop_active:
            # A human is driving via keyboard teleop right now -- stay quiet
            # so we don't fight teleop for control of /cerebri/in/joy.
            return

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ LIDAR sector helpers ------------------
    def _scale_indices(self, num_readings, start_idx_360, end_idx_360):
        start = int(num_readings * start_idx_360 / 360.0)
        end = int(num_readings * end_idx_360 / 360.0)
        return start, end

    def _sector_occupancy_ratio(self, ranges, num_readings, start_idx_360, end_idx_360, threshold):
        start, end = self._scale_indices(num_readings, start_idx_360, end_idx_360)
        sector = ranges[start:end]
        if not sector:
            return 0.0
        close_count = sum(1 for r in sector if math.isfinite(r) and r < threshold)
        return close_count / len(sector)

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        """
        Receives lane boundaries from the camera vector extractor and steers to
        keep the buggy centered between the left/right track edges.

        Uses the "top" (min-y / far) endpoint of each vector -- vector_X[0] --
        rather than the bottom endpoint, so steering reacts slightly ahead of
        the buggy's current position instead of only to what's directly beneath it.

        IMPORTANT: vector_1 is NOT guaranteed to be the left boundary. The
        publisher only assigns vector_1/vector_2 in [left, right] order when
        BOTH sides are detected. When vector_count == 1, whichever single side
        was found (left OR right) is placed in vector_1. So we classify each
        vector by its x-position relative to image center rather than trusting
        the index.
        """
        # Don't fight the zone-guard/parking logic once we've stopped for a QR/building.
        if self.state == self.STATE_IN_ZONE:
            return

        width = message.image_width
        half_width = width / 2.0

        left_x = None
        right_x = None

        if message.vector_count >= 1:
            x = message.vector_1[0].x
            if x < half_width:
                left_x = x
            else:
                right_x = x

        if message.vector_count >= 2:
            x = message.vector_2[0].x
            if x < half_width:
                left_x = x
            else:
                right_x = x

        if left_x is not None and right_x is not None:
            # Both boundaries visible -- true centerline midpoint.
            midpoint = (left_x + right_x) / 2.0
        elif left_x is not None:
            # Only the left boundary visible -- treat the right image edge as a
            # virtual boundary so we bias steering away from the left edge.
            midpoint = (left_x + width) / 2.0
        elif right_x is not None:
            # Only the right boundary visible -- treat the left image edge (x=0)
            # as a virtual boundary so we bias steering away from the right edge.
            midpoint = right_x / 2.0
        else:
            # No boundaries detected at all (e.g. momentary dropout) -- hold the
            # last computed steering instead of snapping back to straight, which
            # would fight whatever curve we were already navigating.
            self.target_turn = self._last_turn
            return

        # Normalize to roughly [-1, 1]. Positive error = track center is to the
        # LEFT of image center -> steer left (positive turn), matching the Joy
        # convention (axes[3] positive = left steer).
        error = (half_width - midpoint) / half_width
        self.target_turn = self._update_lane_pid(error)

    def _update_lane_pid(self, error):
        """PID step for lane centering. Returns a turn command clamped to [TURN_MIN, TURN_MAX]."""
        self._lane_integral = max(min(self._lane_integral + error, LANE_INTEGRAL_LIMIT), -LANE_INTEGRAL_LIMIT)
        derivative = error - self._lane_prev_error
        self._lane_prev_error = error

        turn = (LANE_KP * error) + (LANE_KI * self._lane_integral) + (LANE_KD * derivative)
        turn = max(min(turn, TURN_MAX), TURN_MIN)
        self._last_turn = turn
        return turn

    def lidar_callback(self, message):
        """
        Receives LIDAR range measurements.
        
        GUIDELINE (Obstacle Avoidance & Building Range):
        - `message.ranges` is an array of distances in meters around the buggy.
        - The laser scans cover 360 degrees. Find which indices correspond to the front of the buggy.
        - If a range value in the front sector is below a threshold (e.g. 0.8m), flag an obstacle.
        - Write obstacle avoidance maneuvers (e.g. stop, steer left/right around the block, and merge back).
        - Use LIDAR side-ranges to verify distance to building/QR signs before patient pickup/hospital drop actions.
        """
        ranges = message.ranges
        num_readings = len(ranges)
        if num_readings == 0:
            return

        # --- Zone detection only matters once a matching QR has been seen ---
        if self.state == self.STATE_AWAITING_ZONE:
            left_ratio = self._sector_occupancy_ratio(
                ranges, num_readings, *self.LEFT_SECTOR, threshold=self.BUILDING_DIST_THRESHOLD)
            right_ratio = self._sector_occupancy_ratio(
                ranges, num_readings, *self.RIGHT_SECTOR, threshold=self.BUILDING_DIST_THRESHOLD)

            if left_ratio >= self.BUILDING_OCCUPANCY_RATIO or right_ratio >= self.BUILDING_OCCUPANCY_RATIO:
                self._enter_zone()

    def _enter_zone(self):
        self.state = self.STATE_IN_ZONE
        self.rover_move_manual_mode(0.0, 0.0)
        self.get_logger().info(
            f"In zone for {self.pending_qr_loc} -- stopping for action.")
        self.send_server_update(self.pending_qr_loc)

    def server_communication_callback(self, message):
        """
        Receives coordination commands from the server.
        
        GUIDELINE (Server Communication):
        - Check if the message is destined for the Buggy (`message.dest == 1`).
		- Do not forget to check for ACK messages from server
        - The server communicates mission info in the `message.msg` payload string.
        - Parse server instructions (e.g., patient pickup, target hospitals).
        - Call `self.send_server_update` to report your status when you reach a checkpoint.
        """
        if message.dest != 1:
            return

        self.get_logger().info(f"Received Server Message: {message.msg}")

        raw_msg = message.msg.strip().strip('"')

        # --- Case 1: ACK confirming our own arrival report ---
        if message.ack == 1 and raw_msg == "":
            if message.uid == self.awaiting_ack_uid:
                self.get_logger().info(f"Server ACKed our report (uid={message.uid}).")
                self.awaiting_ack_uid = None
            return

        # --- Case 2: new destination target assigned by the server ---
        if raw_msg != "":
            self._send_ack(message.uid)

            self.current_destination = raw_msg
            self.get_logger().info(f"New destination assigned: {self.current_destination}")

            self.publish_target_destination(self.current_destination)

            self.pending_qr_loc = None
            self.state = self.STATE_EN_ROUTE
            self.target_speed = self.DEFAULT_SPEED

    def publish_target_destination(self, destination_name):
        """Maps a destination name (e.g. 'PATIENT_1', 'HOSPITAL_2') to its sign-board
        letter (per FAQ7) and publishes it to /target_destination for the object
        recognizer node to watch for."""
        sign_letter = self.DESTINATION_TO_SIGN.get(destination_name)
        if sign_letter is None:
            self.get_logger().warn(
                f"No sign-letter mapping for destination '{destination_name}' -- not publishing target.")
            return

        msg = String()
        msg.data = sign_letter
        self.publisher_target_destination.publish(msg)
        self.get_logger().info(f"Published target_destination: '{sign_letter}' (for {destination_name})")

    def send_server_update(self, text_msg):
        """Sends status messages to the server. (Do not forget to send ACK messages to server)"""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = self.own_uid
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)
        self.get_logger().info(
            f"Sent to server -> uid={server_msg.uid} ack={server_msg.ack} msg='{server_msg.msg}'")

        self.awaiting_ack_uid = self.own_uid
        self.own_uid = (self.own_uid + 1) % 256

    def _send_ack(self, uid):
        """
        Sends a bare acknowledgment back to the server for a message it sent us.
        Echoes the server's own uid (not our rolling counter) with ack=1, msg="".
        """
        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        server_msg.uid = uid
        server_msg.ack = 1
        server_msg.msg = ""
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"Sent ACK to server -> uid={uid}")

    def teleop_override_callback(self, message):
        """Tracks whether keyboard teleop currently has manual control."""
        if message.data != self.teleop_active:
            if message.data:
                self.get_logger().info("Teleop override ENGAGED -- pausing autonomous drive commands.")
            else:
                self.get_logger().info("Teleop override RELEASED -- resuming autonomous drive commands.")
        self.teleop_active = message.data

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.
        
        GUIDELINE (Patient/Hospital Identification):
        - Parse the decoded string payload in `message.data` (e.g. "PATIENT_A", "HOSPITAL_B").
        - If it matches your target destination, stop the vehicle close to the building (verify range using LIDAR),
          perform the action (pick patient / drop patient), and communicate the arrival to the server.
        """
        if self.state != self.STATE_EN_ROUTE:
            return  # already busy with obstacle/zone handling -- ignore for now

        self.get_logger().info(f"Heard QR code: {message.data}")

        loc = self._parse_qr_loc(message.data)
        if loc is None:
            return

        if loc != self.current_destination:
            self.get_logger().info(
                f"Ignoring QR for {loc} -- not current target ({self.current_destination}).")
            return

        self.pending_qr_loc = loc
        self.state = self.STATE_AWAITING_ZONE
        self.get_logger().info(f"QR match for target {loc} -- watching for zone.")

    def _parse_qr_loc(self, payload):
        """
        Adapt this to the actual QR payload encoding.
        Placeholder assumes something like "LOC: PATIENT_1".
        """
        try:
            return payload.split(': ')[-1].strip().strip('}')
        except Exception:
            return None

    def sign_board_callback(self, message):
        """
        Receives traffic sign boards (e.g. "TURN_LEFT", "TURN_RIGHT", "TURN_STRAIGHT"
        from the object recognizer).

        GUIDELINE (Sign Board Routing):
        - Use the detected signs to choose the quickest route at intersections.

        Latching behavior: only act on a sign command while we're EN_ROUTE. The
        first one we see moves us into STATE_SIGN_TURNING and latches the
        direction; every sign message after that is ignored until something
        moves us back out of STATE_SIGN_TURNING (the actual turn maneuver and
        the reset back to EN_ROUTE are not implemented yet -- TBD).
        """
        if self.state != self.STATE_EN_ROUTE:
            return
        
        self.get_logger().info(f"Heard Sign Board: {message.data}")

        self.latched_sign_direction = message.data
        self.state = self.STATE_SIGN_TURNING
        self.get_logger().info(f"Latched sign direction: '{self.latched_sign_direction}'.")

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()