#!/usr/bin/env python3
import csv
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import rospy
from mppi_eval_msgs.msg import MPPIEval


def _sanitize_topic(topic: str) -> str:
    s = topic.strip().strip('/')
    s = s.replace('/', '__')
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s)
    return s or 'topic'


def _expand_user(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


@dataclass
class TopicWriter:
    topic: str
    file_path: str
    f: object
    writer: csv.DictWriter
    rows_since_flush: int = 0


class MPPIEvalLogger:
    def __init__(self):
        rospy.init_node('mppi_eval_logger')

        self.log_dir = _expand_user(rospy.get_param('~log_dir', '~/log'))
        self.tag = rospy.get_param('~tag', 'run')
        self.flush_every_n = int(rospy.get_param('~flush_every_n', 20))
        self.topics = rospy.get_param('~topics', [])

        if not isinstance(self.topics, list):
            rospy.logwarn('~topics is not a list; using empty list')
            self.topics = []

        if not self.topics:
            # Common defaults across this repo
            self.topics = [
                '/mppi/eval_info',
                '/mppi_eval',
                '/mppi_ilc_eval',
                '/mppi_ilc_prior_eval',
                '/mppi_tf/eval',
            ]

        run_id = rospy.get_param('/run_id', time.strftime('%Y%m%d_%H%M%S'))
        self.run_dir = os.path.join(self.log_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self._writers: Dict[str, TopicWriter] = {}

        meta_path = os.path.join(self.run_dir, f'meta__{self.tag}.txt')
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f'run_id: {run_id}\n')
            f.write(f'tag: {self.tag}\n')
            f.write(f'topics: {self.topics}\n')
            f.write(f'ros_time_start: {rospy.Time.now().to_sec()}\n')

        for topic in self.topics:
            self._setup_topic(topic)

        rospy.loginfo(f"[mppi_eval_logger] logging to {self.run_dir}")
        rospy.spin()

    def _setup_topic(self, topic: str) -> None:
        safe = _sanitize_topic(topic)
        path = os.path.join(self.run_dir, f'mppi_eval__{self.tag}__{safe}.csv')

        f = open(path, 'w', newline='', encoding='utf-8')
        fieldnames = [
            't',
            'state_cost',
            'global_x', 'global_y', 'global_yaw',
            'cmd_vx', 'cmd_vy', 'cmd_yawrate',
            'cmd_steer_fl', 'cmd_steer_fr', 'cmd_steer_rl', 'cmd_steer_rr',
            'cmd_rotor_fl', 'cmd_rotor_fr', 'cmd_rotor_rl', 'cmd_rotor_rr',
            'calc_time_ms',
            'goal_reached',
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        self._writers[topic] = TopicWriter(topic=topic, file_path=path, f=f, writer=w)

        rospy.Subscriber(topic, MPPIEval, lambda msg, t=topic: self._cb(t, msg), queue_size=10)
        rospy.loginfo(f"[mppi_eval_logger] subscribed: {topic} -> {path}")

    def _cb(self, topic: str, msg: MPPIEval) -> None:
        tw = self._writers.get(topic)
        if tw is None:
            return

        # Prefer message header stamp; fallback to now
        if msg.header.stamp and msg.header.stamp.to_sec() > 0:
            t_sec = msg.header.stamp.to_sec()
        else:
            t_sec = rospy.Time.now().to_sec()

        tw.writer.writerow({
            't': t_sec,
            'state_cost': msg.state_cost,
            'global_x': msg.global_x,
            'global_y': msg.global_y,
            'global_yaw': msg.global_yaw,
            'cmd_vx': msg.cmd_vx,
            'cmd_vy': msg.cmd_vy,
            'cmd_yawrate': msg.cmd_yawrate,
            'cmd_steer_fl': msg.cmd_steer_fl,
            'cmd_steer_fr': msg.cmd_steer_fr,
            'cmd_steer_rl': msg.cmd_steer_rl,
            'cmd_steer_rr': msg.cmd_steer_rr,
            'cmd_rotor_fl': msg.cmd_rotor_fl,
            'cmd_rotor_fr': msg.cmd_rotor_fr,
            'cmd_rotor_rl': msg.cmd_rotor_rl,
            'cmd_rotor_rr': msg.cmd_rotor_rr,
            'calc_time_ms': msg.calc_time_ms,
            'goal_reached': int(msg.goal_reached),
        })

        tw.rows_since_flush += 1
        if self.flush_every_n > 0 and tw.rows_since_flush >= self.flush_every_n:
            try:
                tw.f.flush()
                os.fsync(tw.f.fileno())
            except Exception:
                pass
            tw.rows_since_flush = 0


if __name__ == '__main__':
    try:
        MPPIEvalLogger()
    except rospy.ROSInterruptException:
        pass
