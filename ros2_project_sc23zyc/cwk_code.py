import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
from math import sin, cos
import signal


class ColourNavigator(Node):
    def __init__(self):
        super().__init__('robot_explorer')
        self.bridge = CvBridge()
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10))
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, QoSProfile(depth=10))
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, QoSProfile(depth=10))
        
        self.latest_image = None
        self.found_blue = False
        self.enable_camera_display = False
        self.front_distance = float('inf')
        self.goal_index = 0
        
        self.goals = [
           (-8.15, 1.42, -0.00143), 
           (-1.16, -4.38, -0.00143),
           (6.49, -9.45, -0.00143),
           (-5.44, -10.5, -0.00143), 
        ]
    
    def image_callback(self,msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
            
            if self.enable_camera_display:
                hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([100, 150, 50])
                upper_blue = np.array([140, 255, 255])
                lower_green = np.array([50, 100, 100])
                upper_green = np.array([70, 255, 255])
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([180, 255, 255])

                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                green_mask = cv2.inRange(hsv, lower_green, upper_green)
                red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

                combined_mask = cv2.bitwise_or(blue_mask, green_mask)
                combined_mask = cv2.bitwise_or(combined_mask, red_mask)
                filtered_img = cv2.bitwise_and(self.latest_image, self.latest_image, mask=combined_mask)

                # Display
                if self.latest_image is None:
                    return 
                
                try:
                    cv2.namedWindow('Filtered RGB Detection', cv2.WINDOW_NORMAL)
                    cv2.imshow('Filtered RGB Detection', filtered_img)
                    cv2.resizeWindow('Filtered RGB Detection', 320, 240)
                    cv2.waitKey(1)
                except cv2.error as e:
                    self.get_logger().warn("cv2 error {e}")
            
        except Exception as e: 
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def scan_callback(self, msg):
        center_index = len(msg.ranges) // 2
        window = 10 
        front_ranges = msg.ranges[center_index - window: center_index + window]
        valid_ranges = [r for r in front_ranges if 0.05 < r < 10.0]
        
        if valid_ranges: 
            self.front_distance = min(valid_ranges)
        else:
            self.front_distance = float('inf')
    
    def spin_and_detect(self):
        self.enable_camera_display = True
        self.get_logger().info("spinning ... ")
        
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)

            if self.latest_image is not None:
                break

            if time.time() - start_time > 5.0:
                break
            
        rotate_cmd = Twist()
        rotate_cmd.angular.z = 0.5
        
        wait_time = self.get_clock().now().seconds_nanoseconds()[0]
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.latest_image is not None:
                break
            if self.get_clock().now().seconds_nanoseconds()[0] - wait_time > 5:
                self.get_logger().warn("Camera feed not ready after 5 seconds.")
                break
        
        spin_duration = 8
        start_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        while rclpy.ok():
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            if current_time - start_time > spin_duration:
                break

            self.cmd_vel_pub.publish(rotate_cmd)
            rclpy.spin_once(self, timeout_sec=0.1)

            colour = self.detect_colour()
            if colour:
                self.get_logger().info(f"Detected color: {colour.upper()}")
                if colour == 'blue':
                    self.found_blue = True
                    self.cmd_vel_pub.publish(rotate_cmd)
                    self.center_and_approach_blue()
                    return
        
        self.cmd_vel_pub.publish(Twist())
        self.enable_camera_display = False
    
    def detect_colour(self):
        if self.latest_image is None:
            return None
        
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        # Masks
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        if cv2.countNonZero(blue_mask) > 1000:
            return 'blue'
        elif cv2.countNonZero(green_mask) > 1000:
            return 'green'
        elif cv2.countNonZero(red_mask) > 1000:
            return 'red'
        return None
    
    def center_and_approach_blue(self):
        self.get_logger().info("centering on blue box")
        self.enable_camera_display = True
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_image is None:
                continue
            
            hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            center_x = self.latest_image.shape[1] // 2
            offset = cx - center_x

            if abs(offset) <= 30:
                self.get_logger().info("Blue box centered. Approaching...")
                break
            else:
                rotate = Twist()
                rotate.angular.z = -0.003 * offset
                self.cmd_vel_pub.publish(rotate)

        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.5)

        self.get_logger().info("Moving toward blue box using laser scan...")

        forward = Twist()
        
        while rclpy.ok():      
            rclpy.spin_once(self, timeout_sec= 0.05)
            
            hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            if cv2.countNonZero(blue_mask) < 500:
                self.get_logger().warn("lost sight of blue box")
                break
                  
            if self.front_distance <= 1.0:
                break
            
            if self.front_distance > 2.5:
                forward.linear.x = 0.1
            elif self.front_distance > 1.5:
                forward.linear.x = 0.08
            else:
                forward.linear.x = 0.05
                
            self.cmd_vel_pub.publish(forward)
            
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info(f"Stopped at  approx. {self.front_distance:.2f} meters from blue box.")
        
        self.enable_camera_display = False
        self.shutdown()
             
    def send_next_goal(self):
        if self.goal_index >= len(self.goals) or self.found_blue:
            self.get_logger().info("Task finished.")
            self.shutdown()
            return

        x, y, yaw = [float(v) for v in self.goals[self.goal_index]]
        self.goal_index += 1
        
        self.get_logger().info(f"Navigating to corner {self.goal_index}: (x={x:.1f}, y={y:.1f})")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2.0)

        self.nav_client.wait_for_server()
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected")
            self.send_next_goal()
            return

        self.get_logger().info("Goal accepted.")
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info("Reached goal.")
        self.spin_and_detect()
        
        if not self.found_blue:
            self.send_next_goal()
    
    def shutdown(self):
        if not rclpy.ok():
            return 
        
        self.get_logger().info("Shutting down cleanly...")
        self.cmd_vel_pub.publish(Twist())
        self.enable_camera_display = False
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        

# handling exceptions and such
def main(args=None):
    rclpy.init(args=args)
    node = ColourNavigator()
    
    try:
        start = time.time()
        while time.time() - start < 2.0:   
            rclpy.spin_once(node, timeout_sec=0.1)
        
        node.spin_and_detect()
        if not node.found_blue: 
            node.get_logger().info("🧭 No blue detected. Starting navigation.")
            node.send_next_goal()
        
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
    

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
