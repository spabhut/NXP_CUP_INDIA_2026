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

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Fallback imports for TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None


class ObjectRecognizer(Node):
    """
    ROS 2 Node for traffic sign board detection using SSD MobileNet V2 TFLite model.
    Reads target destination (A, B, C, X, Y, Z), pairs it with adjacent direction arrows
    (Left, Right, Straight), and publishes the decision to /sign_board_detection.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # 1. Dataset Labels (1-indexed mapping matching label_map.pbtxt)
        self.labels = {
            1: 'A', 2: 'B', 3: 'C',
            4: 'Left', 5: 'Right', 6: 'Straight',
            7: 'X', 8: 'Y', 9: 'Z'
        }

        self.destinations = {'A', 'B', 'C', 'X', 'Y', 'Z'}
        self.directions = {'Left', 'Right', 'Straight'}

        # Target Destination (Set dynamically via /target_destination)
        self.target_destination = None
        self.confidence_threshold = 0.50
        self.bridge = CvBridge()

        # 2. Subscriptions & Publishers
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10
        )

        self.subscription_target = self.create_subscription(
            String,
            '/target_destination',
            self.target_callback,
            10
        )

        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10
        )

        self.publisher_debug = self.create_publisher(
            Image,
            '/debug_images/sign_detections',
            10
        )

        # 3. Load TFLite Model
        self.interpreter = None
        if tflite is not None:
            try:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(dir_path, 'sign_detector.tflite')

                if not os.path.exists(model_path):
                    user_home = os.path.expanduser('~')
                    src_fallback = os.path.join(
                        user_home,
                        'cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/sign_detector.tflite'
                    )
                    if os.path.exists(src_fallback):
                        model_path = src_fallback

                if os.path.exists(model_path):
                    self.interpreter = tflite.Interpreter(model_path=model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    self.input_shape = self.input_details[0]['shape']  # [1, 320, 320, 3]
                    self.get_logger().info(f"Loaded TFLite model successfully from: {model_path}")
                else:
                    self.get_logger().error(f"Model file 'sign_detector.tflite' NOT found at {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize TFLite model: {e}")
        else:
            self.get_logger().error("TFLite Runtime not found. Install with: pip install tflite-runtime==2.14.0")

        self.get_logger().info("Object Recognizer Node active. Publish target destination to /target_destination")

    def target_callback(self, msg):
        """Updates active target destination."""
        target = msg.data.strip().upper()
        if target in self.destinations:
            self.target_destination = target
            self.get_logger().info(f"🎯 TARGET SET TO: '{self.target_destination}'")
        else:
            self.get_logger().warn(f"Invalid target '{target}'. Valid targets are: A, B, C, X, Y, Z")

    def camera_image_callback(self, message):
        """Processes incoming camera frames."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None or self.interpreter is None:
            return

        action, debug_frame = self.classify_and_pair(image)

        if action is not None:
            msg = String()
            msg.data = action
            self.publisher_sign.publish(msg)

        if debug_frame is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.publisher_debug.publish(debug_msg)

    def classify_and_pair(self, image):
        """Object detection with strict vertical alignment pairing."""
        h, w, _ = image.shape
        debug_frame = image.copy()

        # Display target status on debug overlay
        status_text = f"Target: {self.target_destination if self.target_destination else 'NONE'}"
        cv2.putText(debug_frame, status_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        try:
            # 1. Preprocess image (320x320 RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(rgb_image, (self.input_shape[2], self.input_shape[1]))
            input_data = np.expand_dims(resized_image, axis=0)

            if self.input_details[0]['dtype'] == np.uint8:
                input_data = input_data.astype(np.uint8)
            else:
                input_data = (input_data.astype(np.float32) - 127.5) / 127.5

            # 2. Run Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # 3. Dynamic Output Parsing
            boxes, classes, scores = None, None, None
            for detail in self.output_details:
                tensor = np.squeeze(self.interpreter.get_tensor(detail['index']))
                if tensor.ndim == 2 and tensor.shape[1] == 4:
                    boxes = tensor
                elif tensor.ndim == 1 and tensor.size > 1:
                    if np.issubdtype(tensor.dtype, np.integer) or np.all(np.mod(tensor, 1) == 0):
                        classes = tensor
                    else:
                        scores = tensor

            if boxes is None or classes is None or scores is None:
                return None, debug_frame

            detected_objects = []

            # 4. Extract Detected Bounding Boxes
            for i in range(len(scores)):
                score = float(scores[i])
                if score >= self.confidence_threshold:
                    class_id = int(classes[i]) + 1  # 1-indexed mapping
                    label = self.labels.get(class_id, f"ID:{class_id}")
                    ymin, xmin, ymax, xmax = boxes[i]

                    pixel_xmin = int(xmin * w)
                    pixel_ymin = int(ymin * h)
                    pixel_xmax = int(xmax * w)
                    pixel_ymax = int(ymax * h)

                    center_x = int((pixel_xmin + pixel_xmax) / 2.0)
                    center_y = int((pixel_ymin + pixel_ymax) / 2.0)

                    detected_objects.append({
                        'label': label,
                        'score': score,
                        'center': (center_x, center_y),
                        'bbox': (pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax)
                    })

                    # Draw detected bounding box
                    cv2.rectangle(debug_frame, (pixel_xmin, pixel_ymin), (pixel_xmax, pixel_ymax), (0, 255, 0), 2)
                    cv2.putText(debug_frame, f"{label} {score:.2f}",
                                (pixel_xmin, max(pixel_ymin - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 5. STRICT VERTICAL ALIGNMENT PAIRING
            if self.target_destination is not None:
                # Locate target destination letter object
                target_obj = next((obj for obj in detected_objects if obj['label'] == self.target_destination), None)

                if target_obj is not None:
                    t_xmin, t_ymin, t_xmax, t_ymax = target_obj['bbox']
                    t_cx, t_cy = target_obj['center']

                    # Allow a small horizontal padding margin (15 pixels)
                    margin = int(w * 0.05)
                    col_left = t_xmin - margin
                    col_right = t_xmax + margin

                    matched_direction = None

                    # Check candidate arrows
                    for obj in detected_objects:
                        if obj['label'] in self.directions:
                            arr_cx, arr_cy = obj['center']

                            # RULE 1: Arrow MUST be vertically below letter (arr_cy > t_cy)
                            # RULE 2: Arrow center_x MUST be horizontally within letter column
                            if arr_cy > t_cy and (col_left <= arr_cx <= col_right):
                                matched_direction = obj['label']
                                break  # Matched exact arrow below letter

                    if matched_direction is not None:
                        action = f"TURN_{matched_direction.upper()}"
                        cv2.putText(debug_frame, f"MATCH: {self.target_destination} -> {action}",
                                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        return action, debug_frame
                    else:
                        # Display NOT FOUND if no arrow is strictly beneath target letter
                        cv2.putText(debug_frame, f"Target '{self.target_destination}' seen | Direction: NOT FOUND",
                                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

        return None, debug_frame

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Node execution error: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()