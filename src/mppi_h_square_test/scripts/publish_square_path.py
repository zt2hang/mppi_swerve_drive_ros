#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

class SquarePathPublisher:
    def __init__(self):
        rospy.init_node('square_path_publisher')

        # Local sliding plan (keeps legacy planners working)
        self.pub_path = rospy.Publisher('/move_base/NavfnROS/plan', Path, queue_size=1, latch=True)
        # Stable full-loop reference path (for ILC-style learning)
        self.pub_full_path = rospy.Publisher('/reference_path_full', Path, queue_size=1, latch=True)
        self.sub_odom = rospy.Subscriber('/groundtruth_odom', Odometry, self.odom_callback)
        
        self.center_square_side_len = rospy.get_param('~side_length', 10.0)
        self.turn_radius = rospy.get_param('~turn_radius', 1.0)
        
        self.L = self.center_square_side_len
        self.straight_len = self.L - 2 * self.turn_radius
        self.arc_len = np.pi / 2 * self.turn_radius
        self.perimeter = 4 * self.straight_len + 4 * self.arc_len
        
        self.errors = []
        self.lap_errors = []
        self.lap_count = 0
        self.last_lap_s = 0.0
        rospy.on_shutdown(self.print_statistics)
        
        self.current_s = 0.0
        self.start_time = rospy.Time.now()
        self.warmup_duration = rospy.Duration(4.0)  # Skip first 2 seconds
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback) # 5Hz
        
        rospy.spin()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Only collect errors after warmup period
        if rospy.Time.now() - self.start_time > self.warmup_duration:
            error = self.calculate_min_distance(x, y)
            self.errors.append(error)
            self.lap_errors.append(error)
        
        # Update current_s estimate
        # We search locally around current_s to find the closest point
        # This handles the loop correctly if we update frequently enough
        best_s = self.current_s
        min_dist = float('inf')
        
        # Search window: scale with perimeter, at least +/- 5.0m for larger paths
        search_window = max(5.0, self.perimeter * 0.1)
        search_range = np.linspace(self.current_s - search_window, self.current_s + search_window, 101)
        
        for s_test in search_range:
            # get_xy_theta returns the actual path coordinates (with y offset)
            px, py, _ = self.get_xy_theta(s_test)
            dist = np.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                best_s = s_test
        
        # Only update if we found a reasonably close point, and ensure forward progress
        if min_dist < 3.0:
            # Prefer forward movement: only go backward if significantly closer
            if best_s >= self.current_s or min_dist < 0.5:
                self.current_s = best_s
            else:
                # Allow small backward adjustment
                self.current_s = max(best_s, self.current_s - 0.5)

        # Check lap completion (use unwrapped s); handle multiple laps if large jump
        while self.current_s - self.last_lap_s >= self.perimeter:
            self.last_lap_s += self.perimeter
            self.report_lap_statistics()

    def timer_callback(self, event):
        self.publish_local_path()
        self.publish_full_reference_path()

    def publish_local_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        # Generate a local path segment from current_s to current_s + lookahead.
        lookahead_dist = 20.0
        ds = 0.02  # higher resolution for sharper corners
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

        # Publish the full closed-loop reference path (stable indexing for learning).
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
        # Shift coordinate to match the centered square definition
        # The path is shifted by +L/2 in Y, so we subtract it to check against centered square
        y -= self.L / 2.0
        
        L = self.L
        R = self.turn_radius
        
        # Define segments (x1, y1, x2, y2)
        segments = [
            (-L/2+R, -L/2, L/2-R, -L/2), # Bottom
            (L/2, -L/2+R, L/2, L/2-R),   # Right
            (L/2-R, L/2, -L/2+R, L/2),   # Top
            (-L/2, L/2-R, -L/2, -L/2+R)  # Left
        ]
        
        # Define arc centers (cx, cy)
        arc_centers = [
            (L/2-R, -L/2+R),  # Bottom-Right
            (L/2-R, L/2-R),   # Top-Right
            (-L/2+R, L/2-R),  # Top-Left
            (-L/2+R, -L/2+R)  # Bottom-Left
        ]
        
        dists = []
        
        # Distance to segments
        for x1, y1, x2, y2 in segments:
            dists.append(self.point_to_segment_dist(x, y, x1, y1, x2, y2))
            
        # Distance to arcs
        for cx, cy in arc_centers:
            dist_to_center = np.hypot(x - cx, y - cy)
            dists.append(abs(dist_to_center - R))
            
        return min(dists)

    def point_to_segment_dist(self, x, y, x1, y1, x2, y2):
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.hypot(x - x1, y - y1)

        # Project point onto line (parameter t)
        t = ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)
        
        # Clamp t to segment [0, 1]
        t = max(0, min(1, t))
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.hypot(x - closest_x, y - closest_y)

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

    def report_lap_statistics(self):
        if not self.lap_errors:
            rospy.loginfo("[SquarePath] Lap %d: no samples", self.lap_count + 1)
            self.lap_count += 1
            return

        errors = np.array(self.lap_errors)
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        self.lap_count += 1
        rospy.loginfo("[SquarePath] Lap %d stats -> RMSE: %.4f m, Mean: %.4f m, Max: %.4f m, Samples: %d",
                      self.lap_count, rmse, mean_error, max_error, len(errors))

        # reset for next lap
        self.lap_errors = []

    def get_xy_theta(self, s):
        x, y, theta = self.get_xy_theta_centered(s)
        y += self.L / 2.0
        return x, y, theta

    def get_xy_theta_centered(self, s):
        """ parametric rounded rectangle (centered at 0,0) """
        # Normalize s to [0, perimeter) for geometric calculation
        # But we need to handle s > perimeter or s < 0 by wrapping
        # However, for the path generation, we want continuous s, so we wrap inside
        
        s_norm = s % self.perimeter
        
        L = self.L
        turn_radius = self.turn_radius
        straight_len = self.straight_len
        arc_len = self.arc_len

        # ... (rest of the logic uses s_norm)
        s = s_norm
        
        # edge 1: bottom side, from center to right corner start
        # s goes from 0 to straight_len/2
        # Wait, let's define s=0 at (0, -L/2) facing East (0 rad)
        
        # Segment 1: Bottom Right Straight half
        # Start (0, -L/2), End (L/2 - R, -L/2)
        seg1_len = straight_len / 2.0
        
        if s < seg1_len:
            x = s
            y = -L/2
            theta = 0.0
            return x, y, theta
        s -= seg1_len
        
        # Segment 2: Bottom Right Corner
        if s < arc_len:
            # Center of corner arc is (L/2 - R, -L/2 + R)
            cx = L/2 - turn_radius
            cy = -L/2 + turn_radius
            # Angle goes from -pi/2 to 0
            ang = -np.pi/2 + (s / turn_radius)
            x = cx + turn_radius * np.cos(ang)
            y = cy + turn_radius * np.sin(ang)
            theta = ang + np.pi/2
            return x, y, theta
        s -= arc_len
        
        # Segment 3: Right Edge
        if s < straight_len:
            x = L/2
            y = -L/2 + turn_radius + s
            theta = np.pi/2
            return x, y, theta
        s -= straight_len
        
        # Segment 4: Top Right Corner
        if s < arc_len:
            cx = L/2 - turn_radius
            cy = L/2 - turn_radius
            ang = 0 + (s / turn_radius)
            x = cx + turn_radius * np.cos(ang)
            y = cy + turn_radius * np.sin(ang)
            theta = ang + np.pi/2
            return x, y, theta
        s -= arc_len
        
        # Segment 5: Top Edge
        if s < straight_len:
            x = L/2 - turn_radius - s
            y = L/2
            theta = np.pi
            return x, y, theta
        s -= straight_len
        
        # Segment 6: Top Left Corner
        if s < arc_len:
            cx = -L/2 + turn_radius
            cy = L/2 - turn_radius
            ang = np.pi/2 + (s / turn_radius)
            x = cx + turn_radius * np.cos(ang)
            y = cy + turn_radius * np.sin(ang)
            theta = ang + np.pi/2
            return x, y, theta
        s -= arc_len
        
        # Segment 7: Left Edge
        if s < straight_len:
            x = -L/2
            y = L/2 - turn_radius - s
            theta = -np.pi/2
            return x, y, theta
        s -= straight_len
        
        # Segment 8: Bottom Left Corner
        if s < arc_len:
            cx = -L/2 + turn_radius
            cy = -L/2 + turn_radius
            ang = np.pi + (s / turn_radius)
            x = cx + turn_radius * np.cos(ang)
            y = cy + turn_radius * np.sin(ang)
            theta = ang + np.pi/2
            return x, y, theta
        s -= arc_len
        
        # Segment 9: Bottom Left Straight half (back to start)
        if s < seg1_len + 0.001: # Add epsilon for float errors
            x = -L/2 + turn_radius + s
            y = -L/2
            theta = 0.0
            return x, y, theta
            
        return 0, -L/2, 0

    def publish_path(self):
        # Deprecated, replaced by publish_local_path
        pass

if __name__ == '__main__':
    try:
        SquarePathPublisher()
    except rospy.ROSInterruptException:
        pass
