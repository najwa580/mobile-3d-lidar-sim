#!/usr/bin/env python3
"""
Nœud d'inférence DQN pour robot mobile
Souscrit à : /scan, /odom
Publie sur  : /cmd_vel
Charge      : model.pth (DQN)
"""

import rclpy
import numpy as np
import math
import threading
import torch
import torch.nn as nn

from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


# ─────────────────────────────────────────────
# Architecture DQN — doit correspondre exactement
# à celle utilisée pendant l'entraînement
# ─────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_size=28, action_size=6):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Nœud d'inférence
# ─────────────────────────────────────────────
class DQNInferenceNode(Node):

    STATE_SIZE     = 28
    ACTION_SIZE    = 6
    SCAN_MAX       = 10.0
    COLLISION_DIST = 0.22
    GOAL_DIST      = 0.40

    _ACTIONS = [
        (0.25,  0.0),   # 0 avance vite
        (0.12,  0.0),   # 1 avance lentement
        (0.08, +0.6),   # 2 tourne gauche
        (0.08, -0.6),   # 3 tourne droite
        (0.0,  +0.8),   # 4 pivot gauche
        (0.0,  -0.8),   # 5 pivot droite
    ]

    def __init__(self, model_path: str, goal_x: float, goal_y: float):
        super().__init__('dqn_inference_node')

        # ── Paramètres ──────────────────────────────────────
        self.goal_x = goal_x
        self.goal_y = goal_y

        # ── État robot ──────────────────────────────────────
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self._last_linear = 0.0

        self._scan_lock = threading.Lock()
        self.scan = None

        # ── Charger le modèle ───────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = DQN(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        # Supporte les deux formats de sauvegarde :
        # torch.save(model.state_dict(), ...) ou torch.save(model, ...)
        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint.to(self.device)

        self.model.eval()
        self.get_logger().info(f"Modèle chargé depuis : {model_path}")
        self.get_logger().info(f"Device : {self.device}")
        self.get_logger().info(f"Goal   : ({goal_x}, {goal_y})")

        # ── Topics ROS2 ─────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, 10)

        # ── Timer d'inférence (10 Hz) ───────────────────────
        self.create_timer(0.1, self._inference_step)

        self.get_logger().info("Nœud d'inférence DQN démarré — en attente des topics...")

    # ─────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────

    def _scan_cb(self, msg):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.SCAN_MAX,
                             posinf=self.SCAN_MAX, neginf=self.SCAN_MAX)
        scan = np.clip(scan, 0.05, self.SCAN_MAX)
        with self._scan_lock:
            self.scan = scan

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ─────────────────────────────────────────────
    # ÉTAT (identique à RobotEnv.get_state)
    # ─────────────────────────────────────────────

    def _get_state(self):
        with self._scan_lock:
            if self.scan is None:
                return None
            scan = self.scan.copy()

        n      = len(scan)
        center = n // 2

        front = scan[max(0, center-10) : center+10]
        left  = scan[center+10 : center+50]
        right = scan[center-50 : center-10]
        back  = np.concatenate([scan[:20], scan[-20:]])

        front_min       = float(np.min(front))        if len(front) > 0 else self.SCAN_MAX
        left_mean_norm  = float(np.mean(np.clip(left,  0.05, self.SCAN_MAX))) / self.SCAN_MAX if len(left)  > 0 else 1.0
        right_mean_norm = float(np.mean(np.clip(right, 0.05, self.SCAN_MAX))) / self.SCAN_MAX if len(right) > 0 else 1.0
        back_mean_norm  = float(np.mean(back)) / self.SCAN_MAX                                if len(back)  > 0 else 1.0
        front_norm      = np.clip(front, 0.05, self.SCAN_MAX) / self.SCAN_MAX

        # Padder/tronquer front_norm à 20 valeurs
        if len(front_norm) < 20:
            front_norm = np.pad(front_norm, (0, 20 - len(front_norm)), constant_values=1.0)
        else:
            front_norm = front_norm[:20]

        dist  = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        angle = math.atan2(math.sin(angle - self.yaw), math.cos(angle - self.yaw))

        vel_norm = np.clip(self._last_linear / 0.25, 0.0, 1.0)

        state = np.array([
            *front_norm,
            min(dist / 15.0, 1.0),
            math.cos(angle),
            math.sin(angle),
            front_min / self.SCAN_MAX,
            left_mean_norm,
            right_mean_norm,
            back_mean_norm,
            vel_norm,
        ], dtype=np.float32)

        return state, dist, front_min

    # ─────────────────────────────────────────────
    # INFÉRENCE (appelée toutes les 100ms)
    # ─────────────────────────────────────────────

    def _inference_step(self):
        result = self._get_state()
        if result is None:
            return

        state, dist, front_min = result

        # ── Vérifier collision ou goal ───────────────────────
        if front_min < self.COLLISION_DIST:
            self.get_logger().warn(f"OBSTACLE PROCHE ({front_min:.2f}m) — arrêt d'urgence !")
            self._publish_cmd(0.0, 0.0)
            return

        if dist < self.GOAL_DIST:
            self.get_logger().info(f"GOAL ATTEINT ! Distance = {dist:.2f}m")
            self._publish_cmd(0.0, 0.0)
            return

        # ── Inférence DQN ────────────────────────────────────
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        action = int(q_values.argmax().item())
        linear, angular = self._ACTIONS[action]
        self._last_linear = linear

        self._publish_cmd(linear, angular)

        self.get_logger().debug(
            f"Action={action} lin={linear:.2f} ang={angular:.2f} "
            f"dist={dist:.2f}m front={front_min:.2f}m"
        )

    def _publish_cmd(self, linear, angular):
        cmd = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    rclpy.init()

    # ── Modifie goal_x et goal_y selon ta mission ────────────
    node = DQNInferenceNode(
        model_path='/home/ahlem/mobile-3d-lidar-sim-main/dqn_model.pth',
        goal_x=5.0,
        goal_y=6.0,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt demandé — robot stoppé.")
        node._publish_cmd(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
