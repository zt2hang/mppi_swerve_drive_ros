#!/usr/bin/env python3
"""
ROS Node for testing MPPI_H with square path tracking.
Acts as a simulator and visualizer.
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist, Quaternion, TransformStamped
from tf.transformations import quaternion_from_euler
import tf2_ros

class SquarePathTrackingTester:
    def __init__(self):
        rospy.init_node('square_path_tracking_tester')
        
        self.center_square_side_len = rospy.get_param('~center_square_side_len', 20.0)
        self.turn_radius = 0.3
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
        
        # Calculate path parameters
        self.L = self.center_square_side_len
        self.straight_len = self.L - 2 * self.turn_radius
        self.arc_len = np.pi / 2 * self.turn_radius
        self.perimeter = 4 * self.straight_len + 4 * self.arc_len
        
        # Initial State
        # Start at bottom edge center
        self.x = 0.0
        self.y = -self.L / 2.0
        self.theta = 0.0
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
        
    def get_xy_theta(self, s):
        """ parametric rounded rectangle """
        s = s % self.perimeter
        L = self.L
        turn_radius = self.turn_radius
        straight_len = self.straight_len
        arc_len = self.arc_len

        # edge 1: bottom side, left->right
        if s < straight_len:
            x = -L/2 + turn_radius + s
            y = -L/2
            theta = 0.0
            return x,y,theta
        s -= straight_len

        # corner 1: bottom-right, CCW quarter arc
        if s < arc_len:
            ang = -np.pi/2 + s/turn_radius
            x =  L/2 - turn_radius + turn_radius*np.cos(ang)
            y = -L/2 + turn_radius + turn_radius*np.sin(ang)
            theta = ang + np.pi/2
            return x,y,theta
        s -= arc_len

        # edge 2: right side, bottom->top
        if s < straight_len:
            x = L/2
            y = -L/2 + turn_radius + s
            theta = np.pi/2
            return x,y,theta
        s -= straight_len

        # corner 2: top-right arc
        if s < arc_len:
            ang = 0 + s/turn_radius
            x = L/2 - turn_radius + turn_radius*np.cos(ang)
            y =  L/2 - turn_radius + turn_radius*np.sin(ang)
            theta = ang + np.pi/2
            return x,y,theta
        s -= arc_len

        # edge 3: top side, right->left
        if s < straight_len:
            x = L/2 - turn_radius - s
            y = L/2
            theta = np.pi
            return x,y,theta
        s -= straight_len

        # corner 3: top-left arc
        if s < arc_len:
            ang = np.pi/2 + s/turn_radius
            x = -L/2 + turn_radius + turn_radius*np.cos(ang)
            y =  L/2 - turn_radius + turn_radius*np.sin(ang)
            theta = ang + np.pi/2
            return x,y,theta
        s -= arc_len

        # edge 4: left side, top->bottom
        if s < straight_len:
            x = -L/2
            y = L/2 - turn_radius - s
            theta = -np.pi/2
            return x,y,theta
        s -= straight_len

        # corner 4: bottom-left arc
        ang = np.pi + s/turn_radius
        x = -L/2 + turn_radius + turn_radius*np.cos(ang)
        y = -L/2 + turn_radius + turn_radius*np.sin(ang)
        theta = ang + np.pi/2
        return x,y,theta

    def setup_plot(self):
        limit = self.L / 2.0 + 5.0
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('MPPI_H Square Path Tracking Test')
        
        # Draw square reference
        half_side = self.L / 2.0
        rect = patches.Rectangle((-half_side, -half_side), self.L, self.L, 
                               linewidth=1, edgecolor='k', facecolor='none', linestyle='--', label='Reference')
        self.ax.add_patch(rect)
        
        self.traj_line, = self.ax.plot([], [], 'r-', label='Trajectory')
        self.robot_point, = self.ax.plot([], [], 'go', label='Robot')
        self.ax.legend()

    def publish_map(self):
        rospy.loginfo("Publishing map...")
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = rospy.Time.now()
        grid.info.resolution = 0.1
        grid.info.width = 400
        grid.info.height = 400
        grid.info.origin.position.x = -20.0
        grid.info.origin.position.y = -20.0
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
        
        # Generate path points
        num_loops = 5
        total_len = self.perimeter * num_loops
        ds = 0.1 # 10cm spacing
        num_points = int(total_len / ds)
        
        for i in range(num_points):
            s = i * ds
            x, y, theta = self.get_xy_theta(s)
            
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            q = quaternion_from_euler(0, 0, theta)
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
        
        # Calculate error (distance to the nearest point on square)
        half_side = self.L / 2.0
        x, y = self.x, self.y
        
        # Distance to edges
        # Edge 0: bottom (y = -half_side)
        dist_bottom = abs(y + half_side) if -half_side <= x <= half_side else np.hypot(x - np.clip(x, -half_side, half_side), y + half_side)
        # Edge 1: right (x = half_side)
        dist_right = abs(x - half_side) if -half_side <= y <= half_side else np.hypot(x - half_side, y - np.clip(y, -half_side, half_side))
        # Edge 2: top (y = half_side)
        dist_top = abs(y - half_side) if -half_side <= x <= half_side else np.hypot(x - np.clip(x, -half_side, half_side), y - half_side)
        # Edge 3: left (x = -half_side)
        dist_left = abs(x + half_side) if -half_side <= y <= half_side else np.hypot(x + half_side, y - np.clip(y, -half_side, half_side))
        
        error = min(dist_bottom, dist_right, dist_top, dist_left)
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
        
        # Publish map and path periodically
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
        tester = SquarePathTrackingTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
