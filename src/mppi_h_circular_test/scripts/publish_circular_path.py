#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

class CircularPathPublisher:
    def __init__(self):
        rospy.init_node('circular_path_publisher')

        # Local sliding plan (keeps legacy planners working)
        self.pub_path = rospy.Publisher('/move_base/NavfnROS/plan', Path, queue_size=1, latch=True)
        # Stable full-loop reference path (for ILC-style learning)
        self.pub_full_path = rospy.Publisher('/reference_path_full', Path, queue_size=1, latch=True)
        self.sub_odom = rospy.Subscriber('/groundtruth_odom', Odometry, self.odom_callback)
        
        self.radius = rospy.get_param('~radius', 5.0)
        # Shift center so path starts at (0,0) and goes East
        self.center_x = 0.0
        self.center_y = self.radius 
        
        self.perimeter = 2 * np.pi * self.radius
        
        self.errors = []
        rospy.on_shutdown(self.print_statistics)
        
        self.current_s = 0.0
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback) # 5Hz
        
        rospy.spin()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        error = self.calculate_min_distance(x, y)
        self.errors.append(error)
        
        # Update current_s estimate
        # We search locally around current_s to find the closest point
        best_s = self.current_s
        min_dist = float('inf')
        
        # Search window: +/- 2.0m
        search_range = np.linspace(self.current_s - 2.0, self.current_s + 2.0, 41)
        
        for s_test in search_range:
            px, py, _ = self.get_xy_theta(s_test)
            dist = np.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                best_s = s_test
                
        self.current_s = best_s

    def timer_callback(self, event):
        self.publish_local_path()
        self.publish_full_reference_path()

    def publish_local_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        # Generate local segment
        lookahead_dist = 20.0
        ds = 0.05
        num_points = int(lookahead_dist / ds)

        for i in range(num_points + 1):
            s = self.current_s + i * ds
            x, y, theta = self.get_xy_theta(s)
            
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, theta))
            path_msg.poses.append(pose)
            
        self.pub_path.publish(path_msg)

    def publish_full_reference_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        ds = 0.05
        num_points = int(np.ceil(self.perimeter / ds))

        for i in range(num_points + 1):
            s = i * ds
            x, y, theta = self.get_xy_theta(s)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, theta))
            path_msg.poses.append(pose)

        self.pub_full_path.publish(path_msg)

    def calculate_min_distance(self, x, y):
        dist_to_center = np.hypot(x - self.center_x, y - self.center_y)
        return abs(dist_to_center - self.radius)

    def print_statistics(self):
        if not self.errors:
            rospy.loginfo("No error data collected.")
            return
            
        errors = np.array(self.errors)
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print("\n" + "="*40)
        print("Path Tracking Statistics")
        print("="*40)
        print(f"Total Samples: {len(errors)}")
        print(f"RMSE:          {rmse:.4f} m")
        print(f"Mean Error:    {mean_error:.4f} m")
        print(f"Max Error:     {max_error:.4f} m")
        print("="*40 + "\n")

    def get_xy_theta(self, s):
        # s is arc length
        # angle = s / R
        # We start at -pi/2 (bottom of circle)
        
        angle = (s / self.radius) - np.pi/2
        
        x = self.center_x + self.radius * np.cos(angle)
        y = self.center_y + self.radius * np.sin(angle)
        theta = angle + np.pi/2
        
        return x, y, theta

if __name__ == '__main__':
    try:
        CircularPathPublisher()
    except rospy.ROSInterruptException:
        pass
