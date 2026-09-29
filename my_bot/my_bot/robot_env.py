import rclpy
import numpy as np
import math
import time
import csv
import threading

from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


class RobotEnv(Node):

    # =========================================================
    # CONSTANTES — adaptées au nouveau robot
    #
    # Nouveau robot (mobile_base.xacro) :
    #   base      : 0.25 × 0.18 × 0.065m
    #   roues     : radius=0.065m, length=0.06m
    #   LiDAR     : range min=0.12m, max=12.0m, sur lidar_link
    #   séparation: (0.18+0.06)/2 + 0.01 = 0.13m × 2 = 0.26m
    # =========================================================
    STATE_SIZE  = 30        # 28 → 30 : +left_min_norm, +right_min_norm
    ACTION_SIZE = 6
    MAX_STEPS   = 200        # timeout par épisode

    # Seuils capteurs — calibrés pour ce robot
    COLLISION_DIST  = 0.22   # m — collision (légèrement > lidar_min=0.12m + marge)
    GOAL_DIST       = 0.40   # m — goal atteint
    DANGER_DIST     = 0.60   # m — zone de danger frontal
    SIDE_DANGER     = 0.30   # m — danger latéral
    SCAN_MAX        = 10.0   # m — portée max utilisée (lidar max=12m, on cap à 10)

    def __init__(self, goal_x=5.0, goal_y=6.0):
        super().__init__("robot_env")

        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        self.goal_x = goal_x
        self.goal_y = goal_y

        self._scan_lock = threading.Lock()
        self._scan_raw  = None
        self.scan       = None

        self.prev_dist    = None
        self._last_linear = 0.0
        self.global_step  = 0

        self.path = []

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Odometry,  "/odom", self._odom_cb, 10)

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self,), daemon=True
        )
        self._spin_thread.start()
        self.get_logger().info("RobotEnv prêt — spin thread démarré")

    # =========================================================
    # CALLBACKS ROS2
    # =========================================================

    def _scan_cb(self, msg):
        scan = np.array(msg.ranges, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.SCAN_MAX,
                             posinf=self.SCAN_MAX, neginf=self.SCAN_MAX)
        scan = np.clip(scan, 0.05, self.SCAN_MAX)
        with self._scan_lock:
            self._scan_raw = scan
            self.scan      = scan.copy()

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.path.append((self.x, self.y))

    # =========================================================
    # HELPERS GÉOMÉTRIQUES
    # =========================================================

    def get_distance(self):
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def get_angle_to_goal(self):
        angle = math.atan2(self.goal_y - self.y, self.goal_x - self.x)
        return math.atan2(
            math.sin(angle - self.yaw),
            math.cos(angle - self.yaw)
        )

    def set_goal(self, gx, gy):
        self.goal_x = gx
        self.goal_y = gy
        self.get_logger().info(f"Nouveau goal : ({gx:.2f}, {gy:.2f})")

    # =========================================================
    # EXTRACTION SECTEURS LIDAR
    # =========================================================

    def _extract_sectors(self, scan):
        """
        360 rays, indexés 0..359
        Index 0/359 = arrière, index 180 = avant (convention LiDAR classique)
        On adapte si nécessaire selon l'orientation réelle du lidar_link.
        """
        n      = len(scan)
        center = n // 2   # index 180 = avant

        # Avant : ±10 rays autour du centre ≈ ±10°
        f_l = max(0, center - 10)
        f_r = min(n, center + 10)
        front = scan[f_l:f_r]

        # Gauche : 10→50 rays à gauche du centre
        left  = scan[center + 10 : center + 50]
        # Droite : 10→50 rays à droite du centre
        right = scan[center - 50 : center - 10]
        # Arrière : premiers + derniers 20 rays
        back  = np.concatenate([scan[:20], scan[-20:]])

        front_min = float(np.min(front)) if len(front) > 0 else self.SCAN_MAX
        left_min  = float(np.min(left))  if len(left)  > 0 else self.SCAN_MAX
        right_min = float(np.min(right)) if len(right) > 0 else self.SCAN_MAX
        back_mean = float(np.mean(back)) if len(back)  > 0 else self.SCAN_MAX

        left_mean_norm  = float(np.mean(np.clip(left,  0.05, self.SCAN_MAX))) / self.SCAN_MAX
        right_mean_norm = float(np.mean(np.clip(right, 0.05, self.SCAN_MAX))) / self.SCAN_MAX
        back_mean_norm  = back_mean / self.SCAN_MAX
        front_norm      = np.clip(front, 0.05, self.SCAN_MAX) / self.SCAN_MAX

        # Min normalisés — info instantanée de danger (pas seulement moyenne)
        left_min_norm  = left_min  / self.SCAN_MAX
        right_min_norm = right_min / self.SCAN_MAX

        return (
            front_norm,
            front_min,
            left_min,
            right_min,
            left_mean_norm,
            right_mean_norm,
            back_mean_norm,
            left_min_norm,
            right_min_norm,
        )

    # =========================================================
    # STATE — 30 dimensions
    # =========================================================
    def get_state(self):
        with self._scan_lock:
            if self.scan is None:
                return np.zeros(self.STATE_SIZE, dtype=np.float32)
            scan = self.scan.copy()

        (
            front_norm,
            front_min,
            _,
            _,
            left_mean_norm,
            right_mean_norm,
            back_mean_norm,
            left_min_norm,
            right_min_norm,
        ) = self._extract_sectors(scan)

        dist  = self.get_distance()
        angle = self.get_angle_to_goal()

        # Vitesse normalisée — base MAX_LINEAR = 0.25 m/s
        vel_norm = np.clip(self._last_linear / 0.25, 0.0, 1.0)

        state = np.array([
            *front_norm,                       # [0..19]  20 valeurs
            min(dist / 15.0, 1.0),             # [20]     distance (normalisé sur 15m)
            math.cos(angle),                   # [21]
            math.sin(angle),                   # [22]
            front_min / self.SCAN_MAX,         # [23]
            left_mean_norm,                    # [24]
            right_mean_norm,                   # [25]
            back_mean_norm,                    # [26]
            vel_norm,                          # [27]
            left_min_norm,                     # [28]  danger latéral instantané gauche
            right_min_norm,                    # [29]  danger latéral instantané droite
        ], dtype=np.float32)

        assert state.shape[0] == self.STATE_SIZE
        return state

    # =========================================================
    # RESET
    # =========================================================
    def reset(self):
        self._publish_cmd(0.0, 0.0)
        self._last_linear = 0.0
        time.sleep(0.5)   # laisser le robot s'arrêter physiquement

        timeout = time.time() + 10.0
        while self.scan is None:
            if time.time() > timeout:
                self.get_logger().error("TIMEOUT : pas de scan LIDAR reçu !")
                break
            time.sleep(0.05)

        self.prev_dist   = self.get_distance()
        self.global_step = 0

        return self.get_state()

    # =========================================================
    # ACTIONS — 6 discrètes
    # Vitesses RÉDUITES pour le nouveau robot (roues 0.065m)
    # =========================================================
    #
    #  0 → avance vite      (0.25 m/s,  0.0 rad/s)
    #  1 → avance lent      (0.12 m/s,  0.0 rad/s)
    #  2 → tourne gauche    (0.08 m/s, +0.6 rad/s)
    #  3 → tourne droite    (0.08 m/s, -0.6 rad/s)
    #  4 → pivot gauche     (0.0  m/s, +0.8 rad/s)
    #  5 → pivot droite     (0.0  m/s, -0.8 rad/s)
    #
    _ACTIONS = [
        (0.25,  0.0),   # 0 avance vite
        (0.12,  0.0),   # 1 avance lentement
        (0.08, +0.6),   # 2 tourne gauche
        (0.08, -0.6),   # 3 tourne droite
        (0.0,  +0.8),   # 4 pivot gauche
        (0.0,  -0.8),   # 5 pivot droite
    ]

    def _publish_cmd(self, linear, angular):
        cmd = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action, episode=0):
        with self._scan_lock:
            if self.scan is None:
                return self.get_state(), -1.0, False
            scan = self.scan.copy()

        linear, angular = self._ACTIONS[action]
        self._last_linear = linear
        self._publish_cmd(linear, angular)

        # Délai légèrement plus long pour ce robot plus lent
        time.sleep(0.15)

        with self._scan_lock:
            if self.scan is not None:
                scan = self.scan.copy()

        (
            _,
            front_min,
            left_min,
            right_min,
            _,
            _,
            _,
            _,
            _,
        ) = self._extract_sectors(scan)

        dist  = self.get_distance()
        angle = self.get_angle_to_goal()

        reward = 0.0
        done   = False

        # ── 1. COLLISION (front + latéral) ────────────────────────
        if min(front_min, left_min, right_min) < self.COLLISION_DIST:
            reward = -10.0
            done   = True
            self._publish_cmd(0.0, 0.0)
            self._log_step(episode, action, reward, dist, front_min, angle, "COLLISION")
            self._finalize(dist)
            return self.get_state(), reward, done

        # ── 2. GOAL ATTEINT ────────────────────────────────────────
        if dist < self.GOAL_DIST:
            reward = +10.0
            done   = True
            self._publish_cmd(0.0, 0.0)
            self._log_step(episode, action, reward, dist, front_min, angle, "GOAL")
            self._finalize(dist)
            return self.get_state(), reward, done

        # ── 3. TIMEOUT ─────────────────────────────────────────────
        self.global_step += 1
        if self.global_step >= self.MAX_STEPS:
            reward = -2.0
            done   = True
            self._publish_cmd(0.0, 0.0)
            self._log_step(episode, action, reward, dist, front_min, angle, "TIMEOUT")
            self._finalize(dist)
            return self.get_state(), reward, done

        # ── 4. ATTRACTION vers le goal (progression) ─────────────
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            reward  += np.clip(progress * 8.0, -1.0, 1.0)

        # ── 5. ORIENTATION vers le goal ──────────────────────────
        reward += 1.0 * math.cos(angle)

        # ── 6. RÉPULSION obstacles (potentiel continu + directionnel) ──
        reward += self._repulsive_reward(front_min, left_min, right_min, action)

        # ── 7. AVANCE LIBRE ET EFFICACE ──────────────────────────
        if front_min > 0.80 and abs(angle) < 0.3 and action == 0:
            reward += 0.8

        # ── 8. PÉNALITÉ ROTATION INUTILE ─────────────────────────
        if front_min > 0.80 and abs(angle) < 0.25 and action in [4, 5]:
            reward -= 0.8

        # ── 9. PÉNALITÉ DE TEMPS ──────────────────────────────────
        reward -= 0.02

        reward = float(np.clip(reward, -10.0, 10.0))

        self._finalize(dist)
        self._log_step(episode, action, reward, dist, front_min, angle)

        return self.get_state(), reward, done

    # =========================================================
    # RÉCOMPENSE RÉPULSIVE — évitement d'obstacles
    # =========================================================
    def _repulsive_reward(self, front_min, left_min, right_min, action):
        """
        Champ de potentiel répulsif : pénalité continue (quadratique)
        proportionnelle à la proximité d'un obstacle, + bonus/malus
        directionnel selon que l'action choisie s'éloigne ou fonce
        dans l'obstacle le plus proche.
        """
        r = 0.0
        safe = self.DANGER_DIST

        # -- Pénalité continue par secteur --
        for d in (front_min, left_min, right_min):
            if d < safe:
                proximity = (safe - d) / safe          # 0 → 1 en s'approchant
                r -= 2.5 * proximity ** 2

        # -- Danger latéral : bonus si on tourne du bon côté --
        left_danger  = left_min  < self.SIDE_DANGER
        right_danger = right_min < self.SIDE_DANGER

        if left_danger and not right_danger:
            # obstacle à gauche → il faut s'écarter vers la droite
            if action in (3, 5):        # tourne droite / pivot droite
                r += 0.8
            elif action in (2, 4):      # tourne/pivot vers l'obstacle
                r -= 0.6
        elif right_danger and not left_danger:
            if action in (2, 4):
                r += 0.8
            elif action in (3, 5):
                r -= 0.6

        # -- Obstacle frontal proche : pénaliser foncer, encourager virer --
        if front_min < self.DANGER_DIST:
            if action == 0:              # avance vite droit sur l'obstacle
                r -= 1.0
            elif action in (2, 3, 4, 5): # correction de trajectoire
                r += 0.3

        return r

    # =========================================================
    # HELPERS
    # =========================================================

    def _finalize(self, dist):
        self.prev_dist = dist

    def _log_step(self, episode, action, reward, dist, front_min, angle, event=""):
        tag = f"[{event}]" if event else ""
        print(
            f"[EP {episode:3d} STEP {self.global_step:3d}] "
            f"A={action} R={reward:+6.3f} D={dist:.2f}m "
            f"F={front_min:.2f}m ang={math.degrees(angle):+5.1f}° {tag}"
        )

    def stop(self):
        self._publish_cmd(0.0, 0.0)

    def save_logs(self, filepath="path.csv"):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(self.path)
        self.get_logger().info(f"Path sauvegardé : {filepath}")
