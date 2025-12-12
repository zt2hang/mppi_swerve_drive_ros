#!/usr/bin/env python3
import csv
import math
import os
import time
import threading
from typing import List, Optional, Tuple

import rospy
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _expand_user(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def _dist(x1, y1, x2, y2) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _path_yaw(pose) -> float:
    q = pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def _curvature_from_three_points(p0, p1, p2) -> float:
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    a = _dist(x0, y0, x1, y1)
    b = _dist(x1, y1, x2, y2)
    c = _dist(x0, y0, x2, y2)
    if a < 1e-9 or b < 1e-9 or c < 1e-9:
        return 0.0
    # 2*Area of triangle via cross product
    area2 = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
    kappa = area2 / (a * b * c)
    return kappa


class PaperLapLogger:
    def __init__(self):
        rospy.init_node('mppi_paper_logger')

        self.log_dir = _expand_user(rospy.get_param('~log_dir', '~/log'))
        self.tag = rospy.get_param('~tag', 'run')
        self.corner_curvature_threshold = float(rospy.get_param('~corner_curvature_threshold', 0.10))
        self.wrap_s_margin = float(rospy.get_param('~wrap_s_margin', 0.5))
        self.closed_path_threshold = float(rospy.get_param('~closed_path_threshold', 0.6))

        self.odom_topic = rospy.get_param('~odom_topic', '/groundtruth_odom')
        self.reference_path_topic = rospy.get_param('~reference_path_topic', '/reference_path_full')

        # Some path publishers republish the same Path at a fixed rate.
        # If we reset lap accounting on every callback, we will never flush per-lap metrics.
        self.reset_on_path_update = bool(rospy.get_param('~reset_on_path_update', False))

        self._lock = threading.Lock()
        self.path: Optional[Path] = None
        self.cum_s: List[float] = []
        self.path_is_closed = False

        self.have_last_s = False
        self.last_s = 0.0
        self.lap = 0

        self.samples_total = 0
        self.samples_corner = 0
        self.samples_straight = 0
        self.sum_lat2_total = 0.0
        self.sum_lat2_corner = 0.0
        self.sum_lat2_straight = 0.0
        self.sum_head2_total = 0.0

        run_id = rospy.get_param('/run_id', time.strftime('%Y%m%d_%H%M%S'))
        self.run_dir = os.path.join(self.log_dir, str(run_id))
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, f'paper_metrics__{self.tag}.csv')

        self.csv_f = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_w = csv.DictWriter(self.csv_f, fieldnames=[
            't', 'tag', 'lap', 'path_length',
            'lat_rmse', 'lat_rmse_straight', 'lat_rmse_corner',
            'head_rmse_deg',
            'samples_total'
        ])
        self.csv_w.writeheader()
        self.csv_f.flush()

        rospy.Subscriber(self.reference_path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=20)

        rospy.loginfo(f"[mppi_paper_logger] writing: {self.csv_path}")
        rospy.spin()

    def _path_cb(self, msg: Path):
        if not msg.poses:
            return

        # Build new state first, then swap atomically.
        pts = msg.poses
        new_cum_s: List[float] = [0.0]
        for i in range(1, len(pts)):
            p0 = pts[i - 1].pose.position
            p1 = pts[i].pose.position
            new_cum_s.append(new_cum_s[-1] + _dist(p0.x, p0.y, p1.x, p1.y))

        new_closed = False
        if len(pts) >= 2:
            p0 = pts[0].pose.position
            pN = pts[-1].pose.position
            new_closed = (_dist(p0.x, p0.y, pN.x, pN.y) < self.closed_path_threshold)

        # Determine whether this is effectively the same path as before.
        with self._lock:
            old_path = self.path
            old_cum_s = self.cum_s

        same_path = False
        if old_path is not None and old_path.poses and len(old_path.poses) == len(msg.poses) and len(old_cum_s) == len(msg.poses):
            o0 = old_path.poses[0].pose.position
            oN = old_path.poses[-1].pose.position
            n0 = msg.poses[0].pose.position
            nN = msg.poses[-1].pose.position

            eps = 1e-3
            if (
                _dist(o0.x, o0.y, n0.x, n0.y) < eps
                and _dist(oN.x, oN.y, nN.x, nN.y) < eps
                and abs(old_cum_s[-1] - new_cum_s[-1]) < 1e-2
            ):
                same_path = True

        with self._lock:
            self.path = msg
            self.cum_s = new_cum_s
            self.path_is_closed = new_closed

            # Only reset when the path actually changes (or if explicitly requested).
            if self.reset_on_path_update or not same_path:
                self.have_last_s = False
                self.last_s = 0.0
                self.lap = 0
                self._reset_accum()

    def _reset_accum(self):
        self.samples_total = 0
        self.samples_corner = 0
        self.samples_straight = 0
        self.sum_lat2_total = 0.0
        self.sum_lat2_corner = 0.0
        self.sum_lat2_straight = 0.0
        self.sum_head2_total = 0.0

    def _closest_idx(self, poses, x: float, y: float) -> int:
        best_i = 0
        best_d = float('inf')
        for i, ps in enumerate(poses):
            p = ps.pose.position
            d = _dist(x, y, p.x, p.y)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            path = self.path
            cum_s = self.cum_s
            path_is_closed = self.path_is_closed

        if path is None or len(cum_s) < 2 or not path.poses:
            return

        # Defensive: ensure cum_s matches poses length
        if len(cum_s) != len(path.poses):
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        idx = self._closest_idx(path.poses, x, y)
        idx = max(0, min(idx, len(cum_s) - 1))
        s = cum_s[idx]
        path_len = cum_s[-1]

        # lap wrap detection (closed-loop only)
        if path_is_closed and path_len > 1e-6:
            if not self.have_last_s:
                self.have_last_s = True
                self.last_s = s
            else:
                if (s + self.wrap_s_margin) < self.last_s:
                    self.lap += 1
                    self._flush_lap(path_len)
                    self._reset_accum()
                    self.last_s = s
                else:
                    self.last_s = s

        # tracking error in path frame
        ps = path.poses[idx].pose
        pyaw = _path_yaw(ps)
        dx = x - ps.position.x
        dy = y - ps.position.y
        lat = (-math.sin(pyaw) * dx + math.cos(pyaw) * dy)
        head = _wrap_pi(yaw - pyaw)

        # curvature estimate
        kappa = 0.0
        if 1 <= idx < (len(path.poses) - 1):
            p0 = path.poses[idx - 1].pose.position
            p1 = path.poses[idx].pose.position
            p2 = path.poses[idx + 1].pose.position
            kappa = _curvature_from_three_points((p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y))

        self.samples_total += 1
        self.sum_lat2_total += lat * lat
        self.sum_head2_total += head * head

        if abs(kappa) >= self.corner_curvature_threshold:
            self.samples_corner += 1
            self.sum_lat2_corner += lat * lat
        else:
            self.samples_straight += 1
            self.sum_lat2_straight += lat * lat

    def _flush_lap(self, path_len: float):
        if self.samples_total < 10:
            return

        n = float(self.samples_total)
        lat_rmse = math.sqrt(self.sum_lat2_total / n)
        head_rmse_deg = math.sqrt(self.sum_head2_total / n) * 180.0 / math.pi

        lat_rmse_corner = math.sqrt(self.sum_lat2_corner / float(self.samples_corner)) if self.samples_corner > 0 else 0.0
        lat_rmse_straight = math.sqrt(self.sum_lat2_straight / float(self.samples_straight)) if self.samples_straight > 0 else 0.0

        self.csv_w.writerow({
            't': rospy.Time.now().to_sec(),
            'tag': self.tag,
            'lap': self.lap,
            'path_length': path_len,
            'lat_rmse': lat_rmse,
            'lat_rmse_straight': lat_rmse_straight,
            'lat_rmse_corner': lat_rmse_corner,
            'head_rmse_deg': head_rmse_deg,
            'samples_total': self.samples_total,
        })
        self.csv_f.flush()


if __name__ == '__main__':
    try:
        PaperLapLogger()
    except rospy.ROSInterruptException:
        pass
