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

# Speeds & Turn Rates (UPDATED VALUES)
LANE_SPEED = 0.6
AWAIT_SPEED = 0.5
TURN_SPEED = 0.3
TURN_OMEGA = 0.8
POLE_DIST_THRESHOLD = 2

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

        # ------------------ 1. Obstacle Avoidance Variables ------------------
        self.obstacle_active = False
        self.avoidance_side = None
        self._prev_obstacle_active = False

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

        # Log obstacle state updates
        if self.obstacle_active != self._prev_obstacle_active:
            self.get_logger().info(f"[OBSTACLE AVOIDANCE] Active: {self.obstacle_active} | Hugging Side: {self.avoidance_side}")
            self._prev_obstacle_active = self.obstacle_active

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
            
        start_90, end_90 = self._scale_indices(num_readings, 85, 95)
        start_270, end_270 = self._scale_indices(num_readings, 265, 275)
        
        sector_90 = self.latest_ranges[start_90:end_90]
        sector_270 = self.latest_ranges[start_270:end_270]
        
        pole_90_detected = any(math.isfinite(r) and r < POLE_DIST_THRESHOLD for r in sector_90)
        pole_270_detected = any(math.isfinite(r) and r < POLE_DIST_THRESHOLD for r in sector_270)
        
        return pole_90_detected and pole_270_detected

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        # 1. Mission Override: If we scanned a QR and are heading to the zone, ignore lane PID and drive straight
        if self.state == self.STATE_IN_ZONE:
            return

        width = message.image_width
        half_width = width / 2.0

        # ------------------ 1. Obstacle Avoidance Interrupt ------------------
        if self.obstacle_active and self.avoidance_side is not None:
            # Force the buggy to hug the designated safe side
            turn = self._steer_pid_side_obstacle(message, width, half_width, self.avoidance_side)
            OBSTACLE_SPEED = 0.2
            self.rover_move_manual_mode(OBSTACLE_SPEED, turn)
            return
        # ---------------------------------------------------------------------

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
            if elapsed > 1.0 and self._check_poles():
                self.get_logger().info("Cross poles detected, returning to lane.")
                self.lane_state = self.STATE_LANE

        else:
            if message.vector_count >= 2:
                self.lane_state = self.STATE_LANE

            elif message.vector_count == 1:
                top_x = message.vector_1[0].x
                bottom_x = message.vector_1[1].x
                
                top_left = top_x < half_width
                bottom_left = bottom_x < half_width

                if self.lane_state == self.STATE_LANE:
                    # Check if this single boundary indicates a sharp turn or not.
                    slope_deg = self._vector_slope_angle_deg(
                        message.vector_1[0].x, message.vector_1[0].y,
                        message.vector_1[1].x, message.vector_1[1].y)
                    if slope_deg < SLOPE_ANGLE_SHARP_DEG:
                        self.turn_dir = 1.0 if (top_x < bottom_x) else -1.0
                        self.lane_state = (
                            self.STATE_SHARP_WAITING if top_left == bottom_left
                            else self.STATE_SHARP_TURNING)

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
            turn = self._steer_pid_side_sign(message, width, half_width, side)
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

    def _steer_pid_side_obstacle(self, message, width, half_width, side):
        """Corrected side-specific lane-following PID."""
        left_x = None
        right_x = None

        # Safely assign vectors without blind overwriting
        vectors_x = []
        if message.vector_count >= 1:
            vectors_x.append(message.vector_1[1].x)
        if message.vector_count >= 2:
            vectors_x.append(message.vector_2[1].x)

        for x in vectors_x:
            if x < half_width:
                # Keep the leftmost line for left_x
                if left_x is None or x < left_x:
                    left_x = x
            else:
                # Keep the rightmost line for right_x
                if right_x is None or x > right_x:
                    right_x = x

        offset = 0.75 * half_width 

        if side == 'LEFT':
            if left_x is not None:
                # INVERTED: To hug left, force the left line to the right side of the screen
                midpoint = left_x - offset 
                error = (half_width - midpoint) / half_width
            else:
                error = 1.0  # Steer HARD left
                
        elif side == 'RIGHT':
            if right_x is not None:
                # INVERTED: To hug right, force the right line to the left side of the screen
                midpoint = right_x + offset 
                error = (half_width - midpoint) / half_width
            else:
                error = -1.0 # Steer HARD right
                
        else:
            return self._last_turn

        return self._update_lane_pid(error)

    def _steer_pid_side_sign(self, message, width, half_width, side):
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

        # ------------------ Obstacle Avoidance Logic ------------------
        # The Buggy's physical orientation maps 180 degrees to dead ahead.
        # Front sectors (detection threshold = 0.8m)
        front_center = self._sector_occupancy_ratio(ranges, num_readings, 165, 195, 0.8)
        front_right  = self._sector_occupancy_ratio(ranges, num_readings, 150, 180, 0.8)
        front_left   = self._sector_occupancy_ratio(ranges, num_readings, 180, 210, 0.8)
        
        # Wide Side regions to detect clearance (threshold = 1.0m)
        side_right = self._sector_occupancy_ratio(ranges, num_readings, 60, 120, 1.0)
        side_left  = self._sector_occupancy_ratio(ranges, num_readings, 240, 300, 1.0)
                
        if not self.obstacle_active:
            # CASE 1: Object Dead Ahead (Blocking the center)
            if front_center >= 0.2:
                self.obstacle_active = True
                # Dynamically choose the path of least resistance
                if front_left < front_right:
                    self.avoidance_side = 'LEFT'
                else:
                    self.avoidance_side = 'RIGHT'
                    
                # ADDED DEBUG LOG:
                self.get_logger().info(f"[OBSTACLE DEBUG] Obstacle detected in CENTER. Chose to avoid by hugging {self.avoidance_side}.")
                    
            # CASE 2: Object predominantly on the right
            elif front_right >= 0.3:
                self.obstacle_active = True
                self.avoidance_side = 'LEFT'  # Hug the left line
                
                # ADDED DEBUG LOG:
                self.get_logger().info("[OBSTACLE DEBUG] Obstacle detected on RIGHT. Chose to avoid by hugging LEFT.")
                
            # CASE 3: Object predominantly on the left
            elif front_left >= 0.3:
                self.obstacle_active = True
                self.avoidance_side = 'RIGHT' # Hug the right line
                
                # ADDED DEBUG LOG:
                self.get_logger().info("[OBSTACLE DEBUG] Obstacle detected on LEFT. Chose to avoid by hugging RIGHT.")

        else:
            # RESET CONDITION: Only disable avoidance when the front is completely clear 
            # (less than 10% occupied) AND the obstacle has passed into our side views.
            front_clear = (front_center < 0.1) and (front_right < 0.1) and (front_left < 0.1)
            obstacle_passed = (side_right > 0.0) or (side_left > 0.0)
            
            if front_clear and obstacle_passed:
                self.obstacle_active = False
                self.avoidance_side = None

        # ------------------ Building Zone Check ------------------
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

        if message.ack == 1 and raw_msg == "":
            if message.uid == self.awaiting_ack_uid:
                self.get_logger().info(f"Server ACKed our report (uid={message.uid}).")
                self.awaiting_ack_uid = None
            return

        if raw_msg != "":
            self._send_ack(message.uid)

            self.current_destination = raw_msg
            self.get_logger().info(f"New destination assigned: {self.current_destination}")

            self.publish_target_destination(self.current_destination)

            self.pending_qr_loc = None
            self.state = self.STATE_EN_ROUTE
            self.target_speed = LANE_SPEED

    def publish_target_destination(self, target_sign):
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

        loc = self._parse_qr_loc(message.data)
        if loc is None:
            return

        expected_loc = self.SIGN_TO_DESTINATION.get(self.current_destination)

        if loc != expected_loc:
            self.get_logger().info(
                f"Ignoring QR for {loc} -- expected {expected_loc} (for target {self.current_destination}).")
            return

        self.pending_qr_loc = message.data
        self.state = self.STATE_AWAITING_ZONE
        self.get_logger().info(f"QR match for target {self.current_destination} ({loc}) -- watching for zone.")

    def _parse_qr_loc(self, payload):
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