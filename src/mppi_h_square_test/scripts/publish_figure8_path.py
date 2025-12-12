#!/usr/bin/env python3
"""
Figure-8 Path Publisher for Slip Compensation Testing

This path is designed to maximize slip effects:
- Continuous turning (no straight sections)
- Alternating turn directions (induces lateral oscillations)
- High curvature sections

The figure-8 shape creates situations where:
1. The robot must constantly adjust for centrifugal-induced slip
2. Slip direction reverses at the crossover point
3. Feedforward compensation should provide clear benefits
"""

import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

class Figure8PathPublisher:
    def __init__(self):
        rospy.init_node('figure8_path_publisher')

        # Local sliding plan (keeps legacy planners working)
        self.pub_path = rospy.Publisher('/move_base/NavfnROS/plan', Path, queue_size=1, latch=True)
        # Stable full-loop reference path (for ILC-style learning)
        self.pub_full_path = rospy.Publisher('/reference_path_full', Path, queue_size=1, latch=True)
        self.sub_odom = rospy.Subscriber('/groundtruth_odom', Odometry, self.odom_callback)
        
        # Figure-8 parameters
        self.radius = rospy.get_param('~radius', 5.0)  # Radius of each lobe
        self.center_gap = rospy.get_param('~center_gap', 0.5)  # Gap at crossover
        
        # Lemniscate of Bernoulli parameters
        # x = a * cos(t) / (1 + sin²(t))
        # y = a * sin(t) * cos(t) / (1 + sin²(t))
        self.a = self.radius * 1.5  # Scale factor
        
        # Path length estimation
        self.path_length = self._compute_path_length()
        
        self.errors = []
        rospy.on_shutdown(self.print_statistics)
        
        self.current_s = 0.0
        self.start_time = rospy.Time.now()
        self.warmup_duration = rospy.Duration(4.0)  # Skip first 2 seconds
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        
        rospy.loginfo(f"[Figure8] Initialized with radius={self.radius}, path_length≈{self.path_length:.2f}m")
        rospy.spin()

    def _compute_path_length(self):
        """Approximate path length by numerical integration"""
        dt = 0.01
        length = 0.0
        prev_x, prev_y = self._lemniscate(0)
        for t in np.arange(dt, 2*np.pi, dt):
            x, y = self._lemniscate(t)
            length += np.hypot(x - prev_x, y - prev_y)
            prev_x, prev_y = x, y
        return length

    def _lemniscate(self, t):
        """Lemniscate of Bernoulli (figure-8 curve)"""
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        denom = 1 + sin_t * sin_t
        x = self.a * cos_t / denom
        y = self.a * sin_t * cos_t / denom
        return x, y

    def _lemniscate_derivative(self, t):
        """Derivative of lemniscate for tangent direction"""
        h = 0.001
        x1, y1 = self._lemniscate(t - h)
        x2, y2 = self._lemniscate(t + h)
        dx = (x2 - x1) / (2 * h)
        dy = (y2 - y1) / (2 * h)
        return dx, dy

    def get_xy_theta(self, s):
        """Get position and heading from arc length parameter"""
        # Convert arc length to parameter t (approximate)
        # Start from t=pi/2 so the path begins at origin (0,0)
        t = (s / self.path_length) * 2 * np.pi + np.pi / 2
        t = t % (2 * np.pi)
        
        x, y = self._lemniscate(t)
        dx, dy = self._lemniscate_derivative(t)
        theta = np.arctan2(dy, dx)
        
        # At t=pi/2, lemniscate gives (0, a/2), so we shift to start at origin
        # The lemniscate at t=pi/2: x=0, y = a*sin(pi/2)*cos(pi/2)/(1+sin^2(pi/2)) = 0
        # Actually at t=0: x=a, y=0. We want to start at origin.
        # Shift the entire path so robot starts at (0,0)
        x0, y0 = self._lemniscate(np.pi / 2)  # Starting point offset
        return x - x0, y - y0, theta

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Only collect errors after warmup period
        if rospy.Time.now() - self.start_time > self.warmup_duration:
            error = self.calculate_min_distance(x, y)
            self.errors.append(error)
        
        # Update current_s by searching locally
        best_s = self.current_s
        min_dist = float('inf')
        search_range = np.linspace(self.current_s - 3.0, self.current_s + 3.0, 61)
        
        for s_test in search_range:
            px, py, _ = self.get_xy_theta(s_test)
            dist = np.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                best_s = s_test
        
        if min_dist < 2.0 and best_s >= self.current_s - 0.5:
            self.current_s = best_s

    def timer_callback(self, event):
        self.publish_local_path()
        self.publish_full_reference_path()

    def publish_local_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        lookahead_dist = 15.0
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
        num_points = int(np.ceil(self.path_length / ds))

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
        """Calculate minimum distance to path"""
        x0, y0 = self._lemniscate(np.pi / 2)  # Starting point offset
        min_dist = float('inf')
        for t in np.linspace(0, 2*np.pi, 200):
            px, py = self._lemniscate(t)
            px -= x0  # Apply same offset as get_xy_theta
            py -= y0
            dist = np.hypot(x - px, y - py)
            min_dist = min(min_dist, dist)
        return min_dist

    def print_statistics(self):
        if not self.errors:
            rospy.loginfo("No error data collected.")
            return
            
        errors = np.array(self.errors)
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print("\n" + "="*50)
        print("Figure-8 Path Tracking Statistics")
        print("="*50)
        print(f"Total Samples: {len(errors)}")
        print(f"RMSE:          {rmse:.4f} m")
        print(f"Mean Error:    {mean_error:.4f} m")
        print(f"Max Error:     {max_error:.4f} m")
        print("="*50 + "\n")

if __name__ == '__main__':
    try:
        Figure8PathPublisher()
    except rospy.ROSInterruptException:
        pass
