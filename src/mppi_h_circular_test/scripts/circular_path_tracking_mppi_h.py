#!/usr/bin/env python3
"""
ROS Node for testing MPPI_H with circular path tracking.
Acts as a simulator and visualizer.
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist, Quaternion, TransformStamped
from tf.transformations import quaternion_from_euler
import tf2_ros

class CircularPathTrackingTester:
    def __init__(self):
        rospy.init_node('circular_path_tracking_tester')
        
        self.center_radius = rospy.get_param('~center_radius', 8.25)
        self.dt = 0.05 # Simulation step (20Hz)
        
        # Publishers
        self.pub_odom = rospy.Publisher('/groundtruth_odom', Odometry, queue_size=1)
        self.pub_path = rospy.Publisher('/move_base/NavfnROS/plan', Path, queue_size=1, latch=True)
        self.pub_map = rospy.Publisher('/map', OccupancyGrid, queue_size=1, latch=True)
        self.pub_costmap = rospy.Publisher('/move_base/local_costmap/costmap', OccupancyGrid, queue_size=1, latch=True)
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Subscribers
        self.sub_cmd = rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback)
        
        # State
        self.x = self.center_radius
        self.y = 0.0
        self.theta = np.pi / 2.0
        self.v = 0.0
        self.w = 0.0
        
        # History
        self.x_hist = []
        self.y_hist = []
        
        # Error tracking
        self.errors = []
        rospy.on_shutdown(self.print_statistics)
        
        self.loop_count = 0
        
        # Publish static map and path once
        self.publish_map()
        self.publish_path()
        
        # Visualization
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        # Timer for simulation loop
        rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        
    def setup_plot(self):
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('MPPI_H Circular Path Tracking Test')
        
        # Draw circle
        theta = np.linspace(0, 2*np.pi, 100)
        x_circ = self.center_radius * np.cos(theta)
        y_circ = self.center_radius * np.sin(theta)
        self.ax.plot(x_circ, y_circ, 'k--', label='Reference')
        
        self.traj_line, = self.ax.plot([], [], 'r-', label='Trajectory')
        self.robot_point, = self.ax.plot([], [], 'go', label='Robot')
        self.ax.legend()

    def publish_map(self):
        rospy.loginfo("Publishing map...")
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = rospy.Time.now()
        grid.info.resolution = 0.1
        grid.info.width = 300
        grid.info.height = 300
        grid.info.origin.position.x = -15.0
        grid.info.origin.position.y = -15.0
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (grid.info.width * grid.info.height)
        self.pub_map.publish(grid)
        
        # Also publish costmap (same empty grid for now)
        self.pub_costmap.publish(grid)

    def publish_path(self):
        rospy.loginfo("Publishing path...")
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        num_points = 2000
        num_loops = 10.5
        for i in range(num_points):
            angle = num_loops * 2 * np.pi * i / num_points
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = self.center_radius * np.cos(angle)
            pose.pose.position.y = self.center_radius * np.sin(angle)
            q = quaternion_from_euler(0, 0, angle + np.pi/2)
            pose.pose.orientation = Quaternion(*q)
            path.poses.append(pose)
            
        self.pub_path.publish(path)

    def cmd_callback(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def timer_callback(self, event):
        # Update state
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta += self.w * self.dt
        
        # Calculate error (distance to the circle)
        current_radius = np.hypot(self.x, self.y)
        error = abs(current_radius - self.center_radius)
        self.errors.append(error)
        
        current_time = rospy.Time.now()
        
        # Publish Odom
        odom = Odometry()
        odom.header.frame_id = "map"
        odom.header.stamp = current_time
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        q = quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation = Quaternion(*q)
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        self.pub_odom.publish(odom)
        
        # Publish TF
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = Quaternion(*q)
        self.tf_broadcaster.sendTransform(t)
        
        # Publish map and path periodically (e.g., every 1 second)
        if self.loop_count % 20 == 0:
            self.publish_map()
            self.publish_path()
        self.loop_count += 1
        
        # Visualization
        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        
        if len(self.x_hist) % 5 == 0:
            self.traj_line.set_data(self.x_hist, self.y_hist)
            self.robot_point.set_data([self.x], [self.y])
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception:
                pass

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

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tester = CircularPathTrackingTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
