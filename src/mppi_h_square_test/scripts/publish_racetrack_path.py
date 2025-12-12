#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler


class RacetrackPathPublisher:
    """Closed-loop 'stadium' racetrack: two straights + two semicircles.

    Publishes:
      - /move_base/NavfnROS/plan : local sliding window segment (legacy planners)
      - /reference_path_full     : full closed-loop path (stable indexing for ILC)
    """

    def __init__(self):
        rospy.init_node('racetrack_path_publisher')

        self.pub_path = rospy.Publisher('/move_base/NavfnROS/plan', Path, queue_size=1, latch=True)
        self.pub_full_path = rospy.Publisher('/reference_path_full', Path, queue_size=1, latch=True)
        self.sub_odom = rospy.Subscriber('/groundtruth_odom', Odometry, self.odom_callback)

        self.straight_length = rospy.get_param('~straight_length', 20.0)  # [m]
        self.radius = rospy.get_param('~radius', 3.0)  # [m]

        self.perimeter = 2.0 * self.straight_length + 2.0 * np.pi * self.radius

        self.current_s = 0.0
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)  # 5Hz

        rospy.loginfo(
            f"[Racetrack] straight_length={self.straight_length}, radius={self.radius}, perimeter≈{self.perimeter:.2f}m")
        rospy.spin()

    def timer_callback(self, _event):
        self.publish_local_path()
        self.publish_full_reference_path()

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Local search around current_s to track progress on the loop
        best_s = self.current_s
        min_dist = float('inf')
        search_window = max(5.0, self.perimeter * 0.08)
        search_range = np.linspace(self.current_s - search_window, self.current_s + search_window, 121)

        for s_test in search_range:
            px, py, _ = self.get_xy_theta(s_test)
            dist = np.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                best_s = s_test

        if min_dist < 3.0:
            self.current_s = best_s

    def publish_local_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        lookahead_dist = 25.0
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

    def get_xy_theta(self, s: float):
        """Parametric racetrack.

        Start at (0, 0) heading +x on the bottom straight, loop counter-clockwise.
        The two straights are at y=0 and y=2R, connected by semicircles centered at
        (L, R) and (0, R).
        """

        s = s % self.perimeter
        L = self.straight_length
        R = self.radius

        # Segment A: bottom straight, from (0,0) -> (L,0)
        if s < L:
            x = s
            y = 0.0
            theta = 0.0
            return x, y, theta
        s -= L

        # Segment B: right semicircle, centered at (L, R), from angle -pi/2 -> +pi/2
        arc_len = np.pi * R
        if s < arc_len:
            ang = -np.pi / 2 + (s / R)
            x = L + R * np.cos(ang)
            y = R + R * np.sin(ang)
            theta = ang + np.pi / 2
            return x, y, theta
        s -= arc_len

        # Segment C: top straight, from (L, 2R) -> (0, 2R)
        if s < L:
            x = L - s
            y = 2.0 * R
            theta = np.pi
            return x, y, theta
        s -= L

        # Segment D: left semicircle, centered at (0, R), from angle +pi/2 -> +3pi/2
        ang = np.pi / 2 + (s / R)
        x = 0.0 + R * np.cos(ang)
        y = R + R * np.sin(ang)
        theta = ang + np.pi / 2
        return x, y, theta


if __name__ == '__main__':
    try:
        RacetrackPathPublisher()
    except rospy.ROSInterruptException:
        pass
