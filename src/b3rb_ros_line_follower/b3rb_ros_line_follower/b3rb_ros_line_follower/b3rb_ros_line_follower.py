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

# Speeds & Turn Rates
LANE_SPEED = 0.75
AWAIT_SPEED = 0.5
TURN_SPEED = 0.5
TURN_OMEGA = 0.8
POLE_DIST_THRESHOLD = 1.5

# Lane-following PID gains.
LANE_KP = 1.2
LANE_KI = 0.0
LANE_KD = 0.3
LANE_INTEGRAL_LIMIT = 1.0

# Sharp-turn detection thresholds
SLOPE_ANGLE_NORMAL_DEG = 60.0   
SLOPE_ANGLE_SHARP_DEG = 45.0    

# Sign-board turn maneuver constants
SIGN_LEFT = 'TURN_LEFT'
SIGN_RIGHT = 'TURN_RIGHT'
SIGN_STRAIGHT = 'TURN_STRAIGHT'

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    """

    # ------------------ Mission states (self.state) ------------------
    STATE_EN_ROUTE = 'EN_ROUTE'
    STATE_AWAITING_ZONE = 'AWAITING_ZONE'
    STATE_IN_ZONE = 'IN_ZONE'

    # ------------------ Lane states (self.lane_state) ------------------
    STATE_LANE = 'LANE'                   
    STATE_SHARP_WAITING = 'SHARP_WAITING' 
    STATE_SHARP_TURNING = 'SHARP_TURNING' 
    STATE_SIGN_WAITING = 'SIGN_WAITING'   
    STATE_SIGN_TURNING = 'SIGN_TURNING'   

    # ------------------ LIDAR sectors ------------------
    LEFT_SECTOR = (210, 330)    
    RIGHT_SECTOR = (30, 150)    

    # ------------------ Thresholds ------------------
    BUILDING_DIST_THRESHOLD = 2      
    BUILDING_OCCUPANCY_RATIO = 0.75     

    # ------------------ Sign-board letter -> Destination name ------------------
    SIGN_TO_DESTINATION = {
        'A': 'PATIENT_1',
        'B': 'PATIENT_2',
        'C': 'PATIENT_3',
        'X': 'HOSPITAL_1',
        'Y': 'HOSPITAL_2',
        'Z': 'HOSPITAL_3',
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
        self.target_speed = LANE_SPEED
        self.target_turn = 0.0
        self.teleop_active = False  

        self._lane_integral = 0.0
        self._lane_prev_error = 0.0
        self._last_turn = 0.0

        self.lane_state = self.STATE_LANE
        self.turn_dir = 0.0   

        self.state = self.STATE_EN_ROUTE

        self.current_destination = "A"   
        self.pending_qr_loc = None
        self.mission_completed = False
        
        self._prev_state = self.state
        self._prev_lane_state = self.lane_state

        self.own_uid = 0              
        self.awaiting_ack_uid = None  

        self.latched_sign_direction = None
        self.latest_ranges = []
        self.pole_timer_start = None

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)
        self.publish_target_destination(self.current_destination)

        self.get_logger().info("Line Follower controller initialized. Safe Drive-Straight Mode active.")

    # ------------------ Drive helpers ------------------
    def publish_drive_commands(self):
        if self.state != self._prev_state or self.lane_state != self._prev_lane_state:
            self.get_logger().info(f"[STATE UPDATE] Mission: {self.state} | Lane: {self.lane_state}")
            self._prev_state = self.state
            self._prev_lane_state = self.lane_state

        if self.teleop_active:
            return

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
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
        
    def _check_poles(self):
        if not self.latest_ranges:
            return False
        
        num_readings = len(self.latest_ranges)
        if num_readings == 0:
            return False
            
        idx_90, _ = self._scale_indices(num_readings, 85, 95)
        idx_270, _ = self._scale_indices(num_readings, 265, 275)
        
        val_90 = self.latest_ranges[idx_90] if idx_90 < num_readings else float('inf')
        val_270 = self.latest_ranges[idx_270] if idx_270 < num_readings else float('inf')
        
        return (math.isfinite(val_90) and val_90 < POLE_DIST_THRESHOLD) or \
               (math.isfinite(val_270) and val_270 < POLE_DIST_THRESHOLD)

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        # 1. Mission Override: If we scanned a QR and are heading to the zone, ignore lane PID and drive straight
        if self.state == self.STATE_IN_ZONE:
            return
        if self.state == self.STATE_AWAITING_ZONE:
            self.rover_move_manual_mode(AWAIT_SPEED, 0.0)
            return

        width = message.image_width
        half_width = width / 2.0

        # 2. Maneuver State Machine Logic
        if self.lane_state == self.STATE_SIGN_WAITING:
            if self._check_poles():
                self.get_logger().info("Poles detected, starting sign turn maneuver.")
                self.lane_state = self.STATE_SIGN_TURNING
                self.pole_timer_start = self.get_clock().now()

        elif self.lane_state == self.STATE_SIGN_TURNING:
            elapsed = 0.0
            if self.pole_timer_start is not None:
                elapsed = (self.get_clock().now() - self.pole_timer_start).nanoseconds / 1e9
            
            # Cooldown of 0.5 seconds prevents triggering on the entrance poles immediately 
            if elapsed > 2.0 and self._check_poles():
                self.get_logger().info("Cross poles detected, returning to lane.")
                self.lane_state = self.STATE_LANE

        else:
            if message.vector_count >= 2:
                self.lane_state = self.STATE_LANE

            elif message.vector_count == 1:
                top_left = message.vector_1[0].x < half_width
                bottom_left = message.vector_1[1].x < half_width

                if self.lane_state == self.STATE_LANE:
                    # Check if this single boundary indicates a sharp turn or not.
                    slope_deg = self._vector_slope_angle_deg(
                        message.vector_1[0].x, message.vector_1[0].y,
                        message.vector_1[1].x, message.vector_1[1].y)
                    if slope_deg < SLOPE_ANGLE_SHARP_DEG:
                        # Near-horizontal -- use the BOTTOM point (the TOP point
                        # is unreliable here) to latch which way to turn. Same
                        # convention as the single-side PID bias below: a left
                        # boundary biases us to turn right, a right boundary
                        # biases us to turn left.
                        self.turn_dir = -1.0 if bottom_left else 1.0
                        self.lane_state = (
                            self.STATE_SHARP_WAITING if top_left == bottom_left
                            else self.STATE_SHARP_TURNING)
                    # slope_deg >= SLOPE_ANGLE_SHARP_DEG (includes the ambiguous
                    # 45-60 band) -- not a sharp turn, stay STATE_LANE, normal
                    # single-side PID handles it.

                elif self.lane_state == self.STATE_SHARP_WAITING and top_left != bottom_left:
                    self.lane_state = self.STATE_SHARP_TURNING

        # 3. Apply Steering and Speed based on Final State
        if self.lane_state == self.STATE_LANE:
            self.target_speed = LANE_SPEED
            turn = self._steer_pid(message, width, half_width)
            
        elif self.lane_state == self.STATE_SHARP_WAITING:
            turn = 0.0  
            self.target_speed = AWAIT_SPEED
            
        elif self.lane_state == self.STATE_SHARP_TURNING:
            turn = self.turn_dir * TURN_OMEGA
            self.target_speed = TURN_SPEED
            
        elif self.lane_state == self.STATE_SIGN_WAITING:
            turn = 0.0  
            self.target_speed = AWAIT_SPEED
            
        elif self.lane_state == self.STATE_SIGN_TURNING:
            # We only arrive in this state for LEFT or RIGHT turns, STRAIGHT is bypassed above
            side = 'LEFT' if self.latched_sign_direction == SIGN_LEFT else 'RIGHT'
            turn = self._steer_pid_side(message, width, half_width, side)
            self.target_speed = TURN_SPEED

        self.rover_move_manual_mode(self.target_speed, turn)

    def _steer_pid(self, message, width, half_width):
        """Normal lane-centering PID: midpoint of the visible boundary/boundaries."""
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
            midpoint = (left_x + right_x) / 2.0
        elif left_x is not None:
            midpoint = (left_x + width) / 2.0
        elif right_x is not None:
            midpoint = right_x / 2.0
        else:
            return self._last_turn

        error = (half_width - midpoint) / half_width
        return self._update_lane_pid(error)

    def _steer_pid_side(self, message, width, half_width, side):
        """Side-specific lane-following PID for intersections. 
        Stays 0.75m distance away from the specific vector line.
        Uses the bottom point of the vector [1] to avoid cross-street interference."""
        left_x = None
        right_x = None

        if message.vector_count >= 1:
            # Anchoring to bottom point using [1] instead of [0] to ignore the horizontal cross-street at the top of the frame.
            x = message.vector_1[1].x
            if x < half_width:
                left_x = x
            else:
                right_x = x

        if message.vector_count >= 2:
            # Anchoring to bottom point using [1] instead of [0] to ignore the horizontal cross-street at the top of the frame.
            x = message.vector_2[1].x
            if x < half_width:
                left_x = x
            else:
                right_x = x

        # Set specific track distance away from line
        offset = 0.75 * half_width 

        # We MUST stick to the designated side. If that side's line is missing 
        # (e.g. went off screen during a sharp turn), force a hard steer in that 
        # direction to find it again. NEVER fall back to tracking the opposite line!
        if side == 'LEFT':
            if left_x is not None:
                midpoint = left_x + offset
                error = (half_width - midpoint) / half_width
            else:
                error = 1.0  # Steer HARD left to reacquire the corner line
        elif side == 'RIGHT':
            if right_x is not None:
                midpoint = right_x - offset
                error = (half_width - midpoint) / half_width
            else:
                error = -1.0 # Steer HARD right to reacquire the corner line
        else:
            return self._last_turn

        return self._update_lane_pid(error)

    def _vector_slope_angle_deg(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx == 0 and dy == 0:
            return 90.0
        return math.degrees(math.atan2(dy, dx))

    def _update_lane_pid(self, error):
        """PID step for lane centering. Returns a turn command clamped to [TURN_MIN, TURN_MAX]."""
        self._lane_integral = max(min(self._lane_integral + error, LANE_INTEGRAL_LIMIT), -LANE_INTEGRAL_LIMIT)
        derivative = error - self._lane_prev_error
        self._lane_prev_error = error

        turn = (LANE_KP * error) + (LANE_KI * self._lane_integral) + (LANE_KD * derivative)
        self._last_turn = turn
        return turn

    def lidar_callback(self, message):
        self.latest_ranges = message.ranges
        ranges = message.ranges
        num_readings = len(ranges)
        if num_readings == 0:
            return

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
            self.target_speed = LANE_SPEED

    def publish_target_destination(self, target_sign):
        """Publishes the target sign-board letter (e.g. 'A', 'X') for the object recognizer."""
        msg = String()
        msg.data = target_sign
        self.publisher_target_destination.publish(msg)
        self.get_logger().info(f"Published target_destination: '{target_sign}'")

    def send_server_update(self, text_msg):
        server_msg = ServerCommunication()
        server_msg.src = 1       
        server_msg.dest = 2      
        server_msg.uid = self.own_uid
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)
        self.get_logger().info(
            f"Sent to server -> uid={server_msg.uid} ack={server_msg.ack} msg='{server_msg.msg}'")

        self.awaiting_ack_uid = self.own_uid
        self.own_uid = (self.own_uid + 1) % 256

    def _send_ack(self, uid):
        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        server_msg.uid = uid
        server_msg.ack = 1
        server_msg.msg = ""
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"Sent ACK to server -> uid={uid}")

    def teleop_override_callback(self, message):
        if message.data != self.teleop_active:
            if message.data:
                self.get_logger().info("Teleop override ENGAGED -- pausing autonomous drive commands.")
            else:
                self.get_logger().info("Teleop override RELEASED -- resuming autonomous drive commands.")
        self.teleop_active = message.data

    def qr_detection_callback(self, message):
        if self.state != self.STATE_EN_ROUTE:
            return  

        self.get_logger().info(f"Heard QR code: {message.data}")

        # Parse string like "{LOC: PATIENT_1}" -> "PATIENT_1"
        loc = self._parse_qr_loc(message.data)
        if loc is None:
            return

        # Convert the current server target ('A') to the expected QR string ('PATIENT_1')
        expected_loc = self.SIGN_TO_DESTINATION.get(self.current_destination)

        if loc != expected_loc:
            self.get_logger().info(
                f"Ignoring QR for {loc} -- expected {expected_loc} (for target {self.current_destination}).")
            return

        # Store the EXACT RAW TEXT from the QR to publish when we enter the zone
        self.pending_qr_loc = message.data
        self.state = self.STATE_AWAITING_ZONE
        self.get_logger().info(f"QR match for target {self.current_destination} ({loc}) -- watching for zone.")

    def _parse_qr_loc(self, payload):
        """Extracts destination like PATIENT_1 from {LOC: PATIENT_1}"""
        try:
            return payload.split(': ')[-1].strip().strip('}')
        except Exception:
            return payload.strip()

    def sign_board_callback(self, message):
        if self.lane_state not in (self.STATE_LANE, self.STATE_SIGN_WAITING):
            return
        
        self.get_logger().info(f"Heard Sign Board: {message.data}")

        self.latched_sign_direction = message.data

        if self.latched_sign_direction == SIGN_STRAIGHT:
            self.lane_state = self.STATE_LANE
            self.get_logger().info("Sign is STRAIGHT. Bypassing wait and directly continuing in lane.")
        else:
            self.lane_state = self.STATE_SIGN_WAITING
            self.get_logger().info(f"Latched sign direction: '{self.latched_sign_direction}'. Waiting for poles.")


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