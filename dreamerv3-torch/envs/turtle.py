import rclpy
import random
import math
import numpy as np
import torch as T
import torch.nn.functional as F
from time import sleep
from rclpy.node import Node
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetEntityState, SetEntityState
import time
import csv
import os
from collections import deque
from datetime import datetime

import gym
from gym import spaces


REACH_TRESHOLD = 0.4
LIDAR_MAX_RANGE = 3.5
COLISION_TRESHOLD = 0.13   # m — min LiDAR (sensor≈centre) to obstacle. Just above the
                           # 0.12 m LiDAR hardware floor (model.sdf); ~0.025 m from the
                           # robot's 0.105 m edge ≈ near-contact. (was 0.2 m)
EASE_DECAY = 0.005
EASE_BEGIN = 0.75
EASE_MIN = 0.01
MEDIUM_RATE = 0.1
MIN_GOAL_DIST = 0.8          # minimum distance from robot reset origin to sampled goal
MAX_GOAL_SPAWN_ATTEMPTS = 10 # retry limit before falling back to farthest candidate


def _pbrs_potential(dist, angle, dist_weight, angle_weight, dist_scale):
    """Potential Φ(s) for potential-based reward shaping (reward_mode='pbrs').

    Φ(s) = -(dist_weight * d_norm + angle_weight * a_norm), where
        d_norm = min(dist / dist_scale, 1.0)   ∈ [0, 1]  (0 at the goal)
        a_norm = (1 - cos(angle)) / 2          ∈ [0, 1]  (0 when facing the goal)
    Φ is ≤ 0 and rises toward 0 as the robot nears and faces the goal, so the
    shaping term F = γ·Φ(s') − Φ(s) rewards goal-directed progress. `dist` is the
    relative goal distance and `angle` the relative goal bearing already computed
    by Env.get_state(). Caller guards against NaN/inf inputs.
    """
    d_norm = min(dist / dist_scale, 1.0)
    a_norm = (1.0 - math.cos(angle)) / 2.0
    return -((dist_weight * d_norm) + (angle_weight * a_norm))


def generate_target_sdf(x, y, z):
        return f"""
        <?xml version='1.0'?>
        <sdf version='1.6'>
        <model name='target_mark'>
            <static>true</static>
            <pose>{x} {y} {z} 0 0 0</pose>
            <link name='link'>
            <visual name='visual'>
                <geometry>
                <cylinder><radius>0.4</radius><length>0.01</length></cylinder>
                </geometry>
                <material>
                <ambient>1 0 0 1</ambient>
                </material>
            </visual>
            </link>
        </model>
        </sdf>
        """

class Env(Node):
    def __init__(self, stage, max_steps, lidar, run_name='baseline', mode='train',
                 odometry_mode='none', device='cpu', resource_logging=False,
                 reward_mode='default', reward_progress_scale=1.0,
                 reward_step_penalty=0.01, reward_turn_penalty=0.01,
                 reward_near_obstacle_scale=0.1, reward_near_obstacle_sigma=0.25,
                 pbrs_scale=1.0, pbrs_distance_weight=1.0, pbrs_angle_weight=0.2,
                 pbrs_distance_scale=5.0, pbrs_gamma=0.997,
                 csv_dir='./csv_logs', plots_dir='./path_plots'):
        super().__init__("trainer_node")

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        # IMU subscription only for full_imu mode — keeps other modes zero-overhead (no 200 Hz callback)
        self.imu_data = None
        if odometry_mode == 'full_imu':
            self.imu_subscription = self.create_subscription(Imu, '/imu', self.imu_callback, 1)
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.get_entity_state_client = self.create_client(GetEntityState, '/demo/get_entity_state')
        self.set_entity_state_client = self.create_client(SetEntityState, '/demo/set_entity_state')
        self.pause_simulation_client = self.create_client(Empty, '/pause_physics')
        self.unpause_simulation_client = self.create_client(Empty, '/unpause_physics')

        self.reset_info()
        self.init_properties(stage, max_steps, lidar, run_name, mode, odometry_mode,
                             device, resource_logging,
                             reward_mode, reward_progress_scale,
                             reward_step_penalty, reward_turn_penalty,
                             reward_near_obstacle_scale, reward_near_obstacle_sigma,
                             pbrs_scale, pbrs_distance_weight, pbrs_angle_weight,
                             pbrs_distance_scale, pbrs_gamma,
                             csv_dir, plots_dir)

    def pause_simulation(self):
        try:
            pause_request = Empty.Request()
            future = self.pause_simulation_client.call_async(pause_request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def unpause_simulation(self):
        try:
            unpause_request = Empty.Request()
            future = self.unpause_simulation_client.call_async(unpause_request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def reset_info(self):
        self.odom_data = None
        self.scan_data = None
        self.imu_data = None

    def init_properties(self, stage, max_steps, lidar, run_name='baseline', mode='train',
                        odometry_mode='none', device='cpu', resource_logging=False,
                        reward_mode='default', reward_progress_scale=1.0,
                        reward_step_penalty=0.01, reward_turn_penalty=0.01,
                        reward_near_obstacle_scale=0.1, reward_near_obstacle_sigma=0.25,
                        pbrs_scale=1.0, pbrs_distance_weight=1.0, pbrs_angle_weight=0.2,
                        pbrs_distance_scale=5.0, pbrs_gamma=0.997,
                        csv_dir='./csv_logs', plots_dir='./path_plots'):
        self.num_states = 14
        self.num_actions = 2
        self.action_upper_bound = .25
        self.action_lower_bound = -.25

        self.step_counter = 0
        self.reset_when_reached = True
        self.reached = False

        self.stage = stage
        self.max_steps = max_steps
        self.lidar = lidar
        self.odometry_mode = odometry_mode

        # Thesis metrics tracking
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.path_length = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.initial_distance = 0.0
        self.min_obstacle_dist = float('inf')
        self.near_collision_count = 0
        self.near_collision_threshold = 0.2   # m — was 0.3. Gates BOTH the near_collisions
                                              # counter (l.331) and the RS-4 reward penalty
                                              # gate (l.639); one variable drives both.
        self.episode_number = 0
        self._mode = mode
        self._run_name = run_name
        self._robot_traj: list = []  # populated each episode in get_state()

        # Odometry ablation: separate from self.prev_x/prev_y used for path-length metrics
        self.prev_odom_x   = 0.0
        self.prev_odom_y   = 0.0
        self.prev_odom_yaw = 0.0
        self._odom_features = None
        # IMU (full_imu mode only): 2D robot-frame linear acceleration from /imu
        self.prev_accel_x = 0.0
        self.prev_accel_y = 0.0

        # CSV Logger — Black-box
        # - success_rate / collision_rate: cumulative over all episodes (overall run performance)
        # - rolling_success_rate_N: rate over the last N episodes (recent learning performance)
        # - resume logic: if the CSV already exists, counters are seeded from it so that
        #   episode numbering and cumulative rates continue smoothly after a restart
        # - mode routing: train env → blackbox_{run_name}.csv, eval env → blackbox_eval_{run_name}.csv
        self.csv_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)
        bb_path = (f'{csv_dir}/blackbox_{run_name}.csv'
                   if mode == 'train'
                   else f'{csv_dir}/blackbox_eval_{run_name}.csv')
        bb_exists = os.path.exists(bb_path)
        self._outcome_history = deque(maxlen=500)
        if bb_exists:
            _ep_num, _ep_cnt, _s_cnt, _c_cnt, _outcomes = 0, 0, 0, 0, []
            try:
                with open(bb_path, 'r', newline='') as _f:
                    for _row in csv.DictReader(_f):
                        try:
                            _ep_num = max(_ep_num, int(_row['episode']))
                            _ep_cnt += 1
                            if _row['outcome'] == 'success':
                                _s_cnt += 1
                            elif _row['outcome'] == 'collision':
                                _c_cnt += 1
                            _outcomes.append(_row['outcome'])
                        except (ValueError, KeyError):
                            continue
            except Exception:
                pass
            self.episode_number  = _ep_num
            self.episode_count   = _ep_cnt
            self.success_count   = _s_cnt
            self.collision_count = _c_cnt
            self._outcome_history = deque(_outcomes[-500:], maxlen=500)
        self._bb_file = open(bb_path, 'a', newline='')
        self._bb_writer = csv.writer(self._bb_file)
        if not bb_exists:
            self._bb_writer.writerow([
                'datetime', 'odometry_mode', 'stage', 'episode', 'outcome',
                'steps_to_goal', 'path_directness',
                'min_obstacle_dist', 'near_collisions',
                'success_rate', 'collision_rate',
                'rolling_success_rate_100', 'rolling_collision_rate_100',
                'rolling_success_rate_500', 'rolling_collision_rate_500',
                'episode_steps',
            ])
        self._bb_file.flush()

        # Planning CSV — A* planned path length for planner_path_efficiency metric
        # mode routing: train → planning_{run_name}.csv, eval → planning_eval_{run_name}.csv
        self._grid_cache: dict = {}
        _pl_name = run_name if mode == 'train' else f'eval_{run_name}'
        self._init_planning_csv(_pl_name)

        # Path plot base dir + subfolder routing: eval plots go to {run_name}_eval/
        self._plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        self._plot_run_name = run_name if mode == 'train' else f'{run_name}_eval'

        # Resource-cost logger (optional, enabled via --resource_logging True)
        # mode routing: train → resource_{run_name}.csv, eval → resource_eval_{run_name}.csv
        self._episode_start_time: float = time.time()
        self._resource_logger = None
        if resource_logging:
            from envs.resource_logger import ResourceLogger
            _resource_run_name = run_name if mode == 'train' else f'eval_{run_name}'
            self._resource_logger = ResourceLogger(
                run_name=_resource_run_name,
                stage=self.stage,
                odometry_mode=self.odometry_mode,
                device=device,
                lidar=self.lidar,
                csv_dir=csv_dir,
                imu_enabled=(self.odometry_mode == 'full_imu'),
            )

        # Reward shaping (opt-in via reward_mode='shaped'; 'default' = original sparse reward)
        self.reward_mode = reward_mode
        self.reward_progress_scale = reward_progress_scale
        self.reward_step_penalty = reward_step_penalty
        self.reward_turn_penalty = reward_turn_penalty
        self.reward_near_obstacle_scale = reward_near_obstacle_scale
        self.reward_near_obstacle_sigma = reward_near_obstacle_sigma
        # Potential-based reward shaping (reward_mode='pbrs')
        self.pbrs_scale = pbrs_scale
        self.pbrs_distance_weight = pbrs_distance_weight
        self.pbrs_angle_weight = pbrs_angle_weight
        self.pbrs_distance_scale = pbrs_distance_scale
        self.pbrs_gamma = pbrs_gamma
        self.prev_distance_to_target = 0.0
        self.prev_angle_to_target = 0.0   # PBRS: previous-step (s_t) angle-to-goal
        self.angle_to_target = 0.0        # set every step in get_state() (s_{t+1})
        self._rc_terminal = self._rc_progress = self._rc_step = 0.0
        self._rc_turn = self._rc_near = self._rc_total = 0.0
        # PBRS per-episode component sums (logged to reward CSV; 0 in default/shaped)
        self._rc_pbrs = self._rc_pbrs_phi_current = self._rc_pbrs_phi_next = 0.0
        self._rc_pbrs_dist = self._rc_pbrs_angle = 0.0

        # Reward-components CSV — per-episode component sums (always written, both modes;
        # in 'default' mode the shaping columns are all 0 and sum_total == sum_terminal).
        # mode routing: train → reward_{run_name}.csv, eval → reward_eval_{run_name}.csv
        rw_path = (f'{csv_dir}/reward_{run_name}.csv' if mode == 'train'
                   else f'{csv_dir}/reward_eval_{run_name}.csv')
        rw_exists = os.path.exists(rw_path)
        self._rw_file = open(rw_path, 'a', newline='')
        self._rw_writer = csv.writer(self._rw_file)
        if not rw_exists:
            self._rw_writer.writerow([
                'datetime', 'reward_mode', 'stage', 'episode', 'outcome',
                'sum_terminal', 'sum_progress', 'sum_step_penalty',
                'sum_turn_penalty', 'sum_near_obstacle', 'sum_total',
                # PBRS columns (appended; 0.0 in default/shaped modes)
                'sum_pbrs', 'sum_pbrs_phi_current', 'sum_pbrs_phi_next',
                'sum_pbrs_distance_component', 'sum_pbrs_angle_component',
            ])
        self._rw_file.flush()

    def odom_callback(self, msg):
        self.odom_data = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def get_state(self, linear_vel, angular_vel):
        self.reset_info()
        rclpy.spin_once(self, timeout_sec=0.5)
        while (self.scan_data is None or self.odom_data is None
               or (self.odometry_mode == 'full_imu' and self.imu_data is None)):
            rclpy.spin_once(self, timeout_sec=0.5)

        turtle_x = self.odom_data.pose.pose.position.x
        turtle_y = self.odom_data.pose.pose.position.y

        q = self.odom_data.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        angle_to_target = math.atan2(self.target_y - turtle_y, self.target_x - turtle_x) - yaw
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))
        self.angle_to_target = angle_to_target   # PBRS: current-state (s_{t+1}) goal bearing

        distance_to_target = math.sqrt((self.target_x - turtle_x) ** 2 + (self.target_y - turtle_y) ** 2)

        lidar_readings = self.scan_data.ranges
        num_samples = self.lidar
        step = (len(lidar_readings) - 1) // (num_samples - 1)
        lidar = [lidar_readings[i * step] if lidar_readings[i * step] != float('inf') else LIDAR_MAX_RANGE for i in range(num_samples)]

        if self.odometry_mode != 'none':
            odom_linear_x  = self.odom_data.twist.twist.linear.x
            odom_angular_z = self.odom_data.twist.twist.angular.z
            dx_w = turtle_x - self.prev_odom_x
            dy_w = turtle_y - self.prev_odom_y
            cos_p = math.cos(self.prev_odom_yaw)
            sin_p = math.sin(self.prev_odom_yaw)
            delta_x_local =  dx_w * cos_p + dy_w * sin_p
            delta_y_local = -dx_w * sin_p + dy_w * cos_p
            delta_yaw = math.atan2(
                math.sin(yaw - self.prev_odom_yaw),
                math.cos(yaw - self.prev_odom_yaw)
            )
            if self.odometry_mode == 'twist':
                raw_odom = [odom_linear_x, odom_angular_z]
            elif self.odometry_mode == 'delta':
                raw_odom = [delta_x_local, delta_y_local, delta_yaw]
            elif self.odometry_mode == 'full':
                raw_odom = [odom_linear_x, odom_angular_z, delta_x_local, delta_y_local, delta_yaw]
            else:  # full_imu = full (5) + 2D linear acceleration from /imu
                accel_x = self.imu_data.linear_acceleration.x
                accel_y = self.imu_data.linear_acceleration.y
                self.prev_accel_x = accel_x
                self.prev_accel_y = accel_y
                raw_odom = [odom_linear_x, odom_angular_z, delta_x_local, delta_y_local, delta_yaw,
                            accel_x, accel_y]
            self._odom_features = F.tanh(T.tensor(raw_odom, dtype=T.float32)).tolist()

        state = lidar + [distance_to_target, angle_to_target, linear_vel, angular_vel]
        state = F.tanh(T.tensor(state)).tolist()

        # Thesis metrics tracking
        self.path_length += math.sqrt((turtle_x - self.prev_x)**2 + (turtle_y - self.prev_y)**2)
        self.prev_x = turtle_x
        self.prev_y = turtle_y
        self._robot_traj.append((turtle_x, turtle_y))

        current_min_lidar = min(lidar)
        if current_min_lidar < self.min_obstacle_dist:
            self.min_obstacle_dist = current_min_lidar
        if current_min_lidar < self.near_collision_threshold:
            self.near_collision_count += 1

        self.prev_odom_x   = turtle_x
        self.prev_odom_y   = turtle_y
        self.prev_odom_yaw = yaw

        return state, turtle_x, turtle_y, lidar

    def respawn_target(self):
        self.despawn_target_mark()
        self.spawn_target_in_environment()

    def reset(self):
        self._episode_start_time = time.time()
        self.step_counter = 0

        if self.reached == False or (self.reached == True and self.reset_when_reached == True):
            req = Empty.Request()
            while not self.reset_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Reset service not available, waiting again...')
            self.reset_client.call_async(req)

        if self.reset_when_reached == False:
            self.pause_simulation()
            self.respawn_target()
            self.unpause_simulation()
        else:
            self.respawn_target()

        self.publish_vel(0.0, 0.0)

        self.scan_data = None
        self.odom_data = None
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)
            sleep(0.1)

        if self.odometry_mode != 'none':
            q0 = self.odom_data.pose.pose.orientation
            self.prev_odom_yaw = math.atan2(
                2 * (q0.w * q0.z + q0.x * q0.y),
                1 - 2 * (q0.y * q0.y + q0.z * q0.z)
            )
            self.prev_odom_x = self.odom_data.pose.pose.position.x
            self.prev_odom_y = self.odom_data.pose.pose.position.y

        state, _, _, _ = self.get_state(0, 0)

        # Thesis metrics reset
        self.start_x = self.odom_data.pose.pose.position.x
        self.start_y = self.odom_data.pose.pose.position.y
        self.prev_x = self.start_x
        self.prev_y = self.start_y
        self.path_length = 0.0
        self.initial_distance = math.sqrt(
            (self.target_x - self.start_x)**2 +
            (self.target_y - self.start_y)**2)
        self.min_obstacle_dist = float('inf')
        self.near_collision_count = 0
        self._robot_traj = [(self.start_x, self.start_y)]

        # Reward-shaping per-episode reset
        self.prev_distance_to_target = self.initial_distance
        # PBRS: seed Φ(s_0) angle from the reset state (get_state(0,0) above set it)
        self.prev_angle_to_target = self.angle_to_target
        self._rc_terminal = self._rc_progress = self._rc_step = 0.0
        self._rc_turn = self._rc_near = self._rc_total = 0.0
        self._rc_pbrs = self._rc_pbrs_phi_current = self._rc_pbrs_phi_next = 0.0
        self._rc_pbrs_dist = self._rc_pbrs_angle = 0.0

        return state

    def publish_vel(self, linear_vel, angular_vel):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)


    def _write_blackbox_csv(self, outcome):
        """Write black-box metrics to CSV.

        Columns written per episode:
          - success_rate / collision_rate: cumulative over the entire run
          - rolling_success_rate_N: computed over the last N outcomes in memory
            (window is seeded from existing CSV rows on startup so rolling rates
            are meaningful immediately after a restart, not just after N new episodes)
        Train env writes to blackbox_{run_name}.csv; eval env to blackbox_eval_{run_name}.csv.
        File routing is done at init — no mode guard needed here.
        """
        self._outcome_history.append(outcome)
        _h = list(self._outcome_history)          # up to 500 most recent outcomes
        _w100 = _h[-100:] if len(_h) >= 100 else _h
        _w500 = _h                                # deque maxlen=500
        rs100 = round(_w100.count('success')   / len(_w100) * 100, 2) if _w100 else 0.0
        rc100 = round(_w100.count('collision') / len(_w100) * 100, 2) if _w100 else 0.0
        rs500 = round(_w500.count('success')   / len(_w500) * 100, 2) if _w500 else 0.0
        rc500 = round(_w500.count('collision') / len(_w500) * 100, 2) if _w500 else 0.0
        self._bb_writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.odometry_mode,
            self.stage,
            self.episode_number,
            outcome,
            self.step_counter if outcome == 'success' else -1,
            min(round(self.initial_distance / self.path_length, 4), 1.0) if self.path_length > 0 else 0.0,
            round(self.min_obstacle_dist, 4),
            self.near_collision_count,
            round(self.success_count / self.episode_count * 100, 2),
            round(self.collision_count / self.episode_count * 100, 2),
            rs100, rc100,
            rs500, rc500,
            self.step_counter,   # episode_steps: total steps this episode (all outcomes)
        ])
        self._bb_file.flush()

    # ── Planning CSV (A* path efficiency) ─────────────────────────────────────

    def _init_planning_csv(self, run_name: str) -> None:
        pl_path = f'{self.csv_dir}/planning_{run_name}.csv'
        pl_exists = os.path.exists(pl_path)
        self._pl_file = open(pl_path, 'a', newline='')
        self._pl_writer = csv.writer(self._pl_file)
        if not pl_exists:
            self._pl_writer.writerow([
                'datetime', 'stage', 'episode', 'outcome',
                'start_x', 'start_y', 'target_x', 'target_y',
                'initial_distance', 'actual_path_length',
                'planned_path_length',
                'planner_path_efficiency', 'planner_path_efficiency_raw',
                'planner_status',
                'planned_path_length_center',
                'planner_path_efficiency_center', 'planner_path_efficiency_center_raw',
                'planner_status_center',
            ])
        self._pl_file.flush()

    def _compute_planned_path(self) -> tuple:
        """Run A* from episode start to both goal region and goal centre.

        Returns (reg_len, reg_wp, reg_status, cen_len, cen_wp, cen_status).
        Status values: ok / no_path / planner_error / unsupported_stage.
        Grid is loaded once; two A* calls share it.
        """
        try:
            from envs.stage_map import get_grid, astar_plan, STAGE_ARENAS, RESOLUTION
            if self.stage not in STAGE_ARENAS:
                s = 'unsupported_stage'
                return None, [], s, None, [], s
            grid, arena = get_grid(self.stage)
            start = (self.start_x, self.start_y)
            goal  = (self.target_x, self.target_y)

            # Region metric — stop at REACH_TRESHOLD (0.4 m), matches success condition
            reg_len, reg_wp = astar_plan(grid, arena, start, goal, REACH_TRESHOLD)
            reg_status = 'ok' if reg_len is not None else 'no_path'

            # Centre metric — stop at RESOLUTION (0.05 m), targets exact goal centre cell
            # RESOLUTION is the safe minimum: worst-case diagonal snap error ≈ 0.035 m < 0.05 m
            cen_len, cen_wp = astar_plan(grid, arena, start, goal, RESOLUTION)
            cen_status = 'ok' if cen_len is not None else 'no_path'

            return reg_len, reg_wp, reg_status, cen_len, cen_wp, cen_status
        except Exception:
            s = 'planner_error'
            return None, [], s, None, [], s

    def _write_planning_csv(self, outcome: str) -> None:
        if self.path_length <= 0:
            reg_len, reg_wp, reg_status = None, [], 'zero_actual_path'
            cen_len, cen_wp, cen_status = None, [], 'zero_actual_path'
        else:
            reg_len, reg_wp, reg_status, cen_len, cen_wp, cen_status = \
                self._compute_planned_path()

        actual = self.path_length

        # Region metric (blank fields for non-ok rows — NaN-friendly in CSV)
        if reg_status == 'ok':
            reg_raw    = round(reg_len / actual, 4)
            reg_capped = round(min(reg_raw, 1.0), 4)
            reg_out    = round(reg_len, 4)
        else:
            reg_raw = reg_capped = reg_out = ''

        # Centre metric
        if cen_status == 'ok':
            cen_raw    = round(cen_len / actual, 4)
            cen_capped = round(min(cen_raw, 1.0), 4)
            cen_out    = round(cen_len, 4)
        else:
            cen_raw = cen_capped = cen_out = ''

        dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._pl_writer.writerow([
            dt_str,
            self.stage,
            self.episode_number,
            outcome,
            round(self.start_x, 4),
            round(self.start_y, 4),
            round(self.target_x, 4),
            round(self.target_y, 4),
            round(self.initial_distance, 4),
            round(actual, 4),
            reg_out, reg_capped, reg_raw, reg_status,
            cen_out, cen_capped, cen_raw, cen_status,
        ])
        self._pl_file.flush()

        # Save path visualization plot (non-critical; never crashes training)
        if reg_wp or cen_wp:
            try:
                from envs.path_viz import save_episode_plot
                save_episode_plot(
                    stage=self.stage,
                    start=(self.start_x, self.start_y),
                    goal=(self.target_x, self.target_y),
                    astar_waypoints=reg_wp,
                    astar_center_waypoints=cen_wp,
                    robot_traj=list(self._robot_traj),
                    episode=self.episode_number,
                    outcome=outcome,
                    run_name=self._plot_run_name,
                    efficiency=reg_capped if reg_status == 'ok' else '',
                    efficiency_center=cen_capped if cen_status == 'ok' else '',
                    plots_dir=self._plots_dir,
                    timestamp=dt_str,
                )
            except Exception:
                pass

    def get_reward_and_done(self, turtle_x, turtle_y, target_x, target_y, lidar_32, ang_vel_cmd=0.0):
        reward = 0
        done = False

        distance = np.sqrt((turtle_x - target_x)**2 + (turtle_y - target_y)**2)

        if distance < REACH_TRESHOLD:
            self.reached = True
            done = True
            reward = 100
            print('[log] Turtlebot3 reached target')
            self.episode_count += 1
            self.episode_number += 1
            self.success_count += 1
            self.log_episode_number = float(self.episode_number)
            self.log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_success = 1
            self.log_steps_to_goal = self.step_counter
            self.log_path_directness = min(round(self.initial_distance / self.path_length, 4), 1.0) if self.path_length > 0 else 0.0
            self.log_min_obstacle_dist = round(self.min_obstacle_dist, 4)
            self.log_near_collisions = self.near_collision_count
            self.log_success_rate = round(self.success_count / self.episode_count * 100, 2)
            self.log_collision_rate = round(self.collision_count / self.episode_count * 100, 2)
            self._write_blackbox_csv('success')
            self._write_planning_csv('success')
            if self._resource_logger is not None:
                self._resource_logger.log_episode(self.episode_number, self._episode_start_time)

        elif np.min(lidar_32) < COLISION_TRESHOLD:
            self.reached = False
            done = True
            reward = -10
            print('[log] Turtlebot3 colided with object')
            self.episode_count += 1
            self.episode_number += 1
            self.collision_count += 1
            self.log_episode_number = float(self.episode_number)
            self.log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_success = 0
            self.log_steps_to_goal = -1
            self.log_path_directness = min(round(self.initial_distance / self.path_length, 4), 1.0) if self.path_length > 0 else 0.0
            self.log_min_obstacle_dist = round(self.min_obstacle_dist, 4)
            self.log_near_collisions = self.near_collision_count
            self.log_success_rate = round(self.success_count / self.episode_count * 100, 2)
            self.log_collision_rate = round(self.collision_count / self.episode_count * 100, 2)
            self._write_blackbox_csv('collision')
            self._write_planning_csv('collision')
            if self._resource_logger is not None:
                self._resource_logger.log_episode(self.episode_number, self._episode_start_time)

        elif self.step_counter >= (self.max_steps - 1):
            self.reached = False
            done = True
            reward = -10
            print('[log] Turtlebot3 reached step limit')
            self.episode_count += 1
            self.episode_number += 1
            self.log_episode_number = float(self.episode_number)
            self.log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_success = 0
            self.log_steps_to_goal = -1
            self.log_path_directness = min(round(self.initial_distance / self.path_length, 4), 1.0) if self.path_length > 0 else 0.0
            self.log_min_obstacle_dist = round(self.min_obstacle_dist, 4)
            self.log_near_collisions = self.near_collision_count
            self.log_success_rate = round(self.success_count / self.episode_count * 100, 2)
            self.log_collision_rate = round(self.collision_count / self.episode_count * 100, 2)
            self._write_blackbox_csv('timeout')
            self._write_planning_csv('timeout')
            if self._resource_logger is not None:
                self._resource_logger.log_episode(self.episode_number, self._episode_start_time)

        # ── Optional reward shaping (additive; default mode is a no-op) ──────
        terminal_component = reward  # 0 on normal steps, +100/-10 on terminal
        progress = step_pen = turn_pen = near_pen = 0.0
        r_pbrs = phi_current = phi_next = pbrs_dist_c = pbrs_angle_c = 0.0
        if self.reward_mode == 'shaped':
            progress = self.reward_progress_scale * (self.prev_distance_to_target - distance)
            step_pen = -self.reward_step_penalty
            turn_pen = -self.reward_turn_penalty * abs(ang_vel_cmd)
            d_min = float(np.min(lidar_32))
            if d_min < self.near_collision_threshold:
                near_pen = -self.reward_near_obstacle_scale * math.exp(
                    -d_min / self.reward_near_obstacle_sigma)
            reward += progress + step_pen + turn_pen + near_pen
        elif self.reward_mode == 'pbrs':
            # Potential-based shaping: F = pbrs_scale · (γ·Φ(s_{t+1}) − Φ(s_t)).
            # prev_* describe s_t; `distance` / `self.angle_to_target` describe s_{t+1}
            # (same relative goal distance/angle already in the observation).
            prev_d, prev_a = self.prev_distance_to_target, self.prev_angle_to_target
            curr_d, curr_a = distance, self.angle_to_target
            # Guard: skip shaping this step (no crash) if any value is NaN/inf
            if all(np.isfinite(v) for v in (curr_d, curr_a, prev_d, prev_a)):
                phi_current = _pbrs_potential(prev_d, prev_a, self.pbrs_distance_weight,
                                              self.pbrs_angle_weight, self.pbrs_distance_scale)
                phi_next = _pbrs_potential(curr_d, curr_a, self.pbrs_distance_weight,
                                           self.pbrs_angle_weight, self.pbrs_distance_scale)
                r_pbrs = self.pbrs_scale * (self.pbrs_gamma * phi_next - phi_current)
                if not np.isfinite(r_pbrs):
                    r_pbrs = 0.0
                reward += r_pbrs
                # Diagnostic-only component split (summed into the reward CSV)
                pbrs_dist_c = min(curr_d / self.pbrs_distance_scale, 1.0)
                pbrs_angle_c = (1.0 - math.cos(curr_a)) / 2.0
        self.prev_distance_to_target = distance
        self.prev_angle_to_target = self.angle_to_target

        self._rc_terminal += terminal_component
        self._rc_progress += progress
        self._rc_step     += step_pen
        self._rc_turn     += turn_pen
        self._rc_near     += near_pen
        self._rc_total    += reward
        self._rc_pbrs             += r_pbrs
        self._rc_pbrs_phi_current += phi_current
        self._rc_pbrs_phi_next    += phi_next
        self._rc_pbrs_dist        += pbrs_dist_c
        self._rc_pbrs_angle       += pbrs_angle_c

        if done:
            outcome = ('success' if distance < REACH_TRESHOLD
                       else 'collision' if np.min(lidar_32) < COLISION_TRESHOLD
                       else 'timeout')
            # Surfaced to simulate() via Turtle.step info for correct eval-rate
            # accounting (reward-mode-independent; separates timeout from collision).
            self.last_outcome = outcome
            self._rw_writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                self.reward_mode, self.stage, self.episode_number, outcome,
                round(self._rc_terminal, 4), round(self._rc_progress, 4),
                round(self._rc_step, 4), round(self._rc_turn, 4),
                round(self._rc_near, 4), round(self._rc_total, 4),
                round(self._rc_pbrs, 4), round(self._rc_pbrs_phi_current, 4),
                round(self._rc_pbrs_phi_next, 4), round(self._rc_pbrs_dist, 4),
                round(self._rc_pbrs_angle, 4),
            ])
            self._rw_file.flush()

        return reward, done

    def spawn_target_in_environment(self):
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.target_x, self.target_y = self.generate_random_target_position()
        fixed_z = 0.005  # half of cylinder length so disk sits flush on floor

        request = SpawnEntity.Request()
        request.name = 'target_mark'
        request.xml = generate_target_sdf(self.target_x, self.target_y, fixed_z)

        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.handle_spawn_result(future, fixed_z)

        sleep(0.5)
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def despawn_target_mark(self):
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DeleteEntity service not available, waiting again...')

        request = DeleteEntity.Request()
        request.name = 'target_mark'
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.handle_despawn_result(future)

    def step(self, action):
        action = action['action']

        rclpy.spin_once(self, timeout_sec=0.5)
        self.publish_action(action)
        rclpy.spin_once(self, timeout_sec=0.5)

        obs, turtle_x, turtle_y, lidar32 = self.get_state(action[0], action[1])

        self.step_counter += 1

        reward, done = self.get_reward_and_done(
            turtle_x, turtle_y, self.target_x, self.target_y, lidar32, float(action[1]))

        return reward, done, obs

    def _sample_target_position(self):
        if self.stage == 1:
            return (random.uniform(-1.90, 1.90), random.uniform(-1.90, 1.90))
        elif self.stage == 2:
            area = np.random.randint(0, 5)
            if area == 0:
                return (random.uniform(-1.90, 1.90), random.uniform(-1.5, -1.9))
            elif area == 1:
                return (random.uniform(-1.90, 1.90), random.uniform(1.5, 1.9))
            elif area == 2:
                return (random.uniform(1.5, 1.9), random.uniform(-1.90, 1.90))
            elif area == 3:
                return (random.uniform(-1.5, -1.9), random.uniform(-1.90, 1.90))
            elif area == 4:
                return (random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7))
        elif self.stage == 3:
            points = [(0.5, 1), (0, 0.8), (-.4, 0.5), (-1.5, 1.5),
                        (-1.7, 0), (-1.5, -1.5), (1.7, 0), (1.7, -.8),
                        (1.7, -1.7), (0.8, -1.5), (0, -1), (-.4, -2),
                        (-1.8, -1.8), (-1.8, -0.5), (-1.8, -1), (-1.8, -1.5),
                        (-1.6, -1.6), (-1.5, -1.5), (-1.2, -1.2), (-1.3, -1.3),
                        (1.5, 1.5), (1.5, 1), (1, 1), (0, 1), (.7, 1.7), (-1., 1.7),
                        (-1.5, 1.7), (1.5, 0)]
            return random.choice(points)
        elif self.stage == 4:
            points = [
                (0.7, 0), (0.7, 0.5), (0.7, 1.0), (0.7, 1.5), (0.7, 2.0),
                (0.5, -0.5), (0.5, -1.), (0.5, -1.5), (0.5, -2.0),
                (0.0, 1.0), (0.0, -1.0), (0.0, -1.5), (0.0, -2.0),
                (-0.7, 0.0), (-0.7, 0.5), (-0.7, 1.0), (-0.7, -0.5), (-0.7, -1.0),
                (-0.7, 2.0), (-0.2, 2.0), (-2.0, 0), (-2.0, -0.5), (-2.0, 0.5), (-2.0, 1.0),
                (-2.0, 1.5), (-2.0, 2.0), (-2.0, -1.0), (-2.0, -2.0), (-1.5, -2.0), (-1, -2.0),
                (2.0, 2.0), (2.0, 1.5), (2.0, 1.2), (1.6, 2.0), (1.6, 1.5), (1.6, 1.2),
                (2.0, 0.0), (1.5, 0.0), (2.0, -0.5), (1.5, -0.5), (2.0, -1.0), (1.5, -1.0), (2.0, -2.0), (1.5, -2.0), (1.0, -2.0)
            ]
            return random.choice(points)
        elif self.stage == 5:
            points = [
                (3.0, 3.0), (3.0, 2.5), (3.0, 1.5), (3.0, 1.0), (3.0, 0.5), (3.0, 0.0),
                (3.0, -3.0), (3.0, -2.5), (3.0, -2.0), (3.0, -1.5), (3.0, -1.0), (3.0, -0.5),
                (2.5, 3.0), (2.0, 3.0), (1.5, 3.0), (1.0, 3.0), (0.5, 3.0), (0.0, 3.0),
                (-2.5, 3.0), (-2.0, 3.0), (-1.5, 3.0), (-1.0, 3.0), (-0.5, 3.0), (-3.0, 3.0),
                (0.0, 2.5), (0.0, 2), (0.5, 2.5), (0.5, 2), (-0.5, 2.5), (-0.5, 2),
                (1.7, 2), (1.7, 1.5), (1.7, 1.0), (1.7, 0.0), (1.7, -0.5), (1.7, -1.0),
                (2.0, 0.0), (2.0, -0.5), (2.0, -1.0), (2.5, 0.0), (2.5, -0.5), (2.5, -1.0),
                (-2.0, 0.0), (-2.0, -0.5), (-2.0, -1.0), (-2.5, 0.0), (-2.5, -0.5), (-2.5, -1.0),
                (0.0, 1.0), (0.0, -1.0), (0.8, 0.0), (-0.8, 0.0), (1.0, 1.0), (-1.0, 1.0), (0.6, -1.0), (-1.0, -1.0),
                (2.5, -3.0), (2.0, -3.0), (1.5, -3.0), (1.0, -3.0), (0.5, -3.0), (0.0, -3.0), (3.0, -3.0),
                (-2.5, -3.0), (-2.0, -3.0), (-1.5, -3.0), (-1.0, -3.0), (-0.5, -3.0), (-3.0, -3.0),
                (0.5, -1.0), (0.1, -1.0), (0.5, -1.5), (0.1, -1.5), (0.5, -2.5), (0.1, -2.5),
                (1.0, -2.5), (1.5, -2.5), (2.0, -2.5), (2.5, -2.5), (3.0, -2.5)
            ]
            return random.choice(points)
        elif self.stage == 6:
            points = [
                (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1),
                (1.5, 0), (0, -1.5), (-1.5, 0), (1.5, -1.5),
                (6.5, 3), (6.5, 2.5), (6.5, 2.0), (6.5, 1.5), (6.5, 1.0), (6.5, 0.5), (6.5, 0),
                (6.5, -3), (6.5, -2.5), (6.5, -2.0), (6.5, -1.5), (6.5, -1.0), (6.5, -0.5),
                (6, 3), (6, 2.5), (5.5, 3), (5.5, 2.5), (5, 3), (5, 2.5), (4.5, 3), (4.5, 2.5),
                (4, 3), (4, 2.5), (4, 2), (4, 1.5), (3.5, 3), (3.5, 2.5), (3.5, 2), (3.5, 1.5),
                (3, 3), (3, 2.5), (3, 2), (3, 1.5), (2.5, 3), (2.5, 2.5), (2.5, 2), (2.5, 1.5),
                (2, 3), (2, 2.5), (2, 2), (2, 1.5), (1.5, 3), (1.5, 2.5), (1.5, 2),
                (1, 3), (1, 2.5), (1, 2), (0.5, 3), (0.5, 2.5), (0.5, 2),
                (0, 3), (0, 2.5), (0, 2), (-0.5, 3), (-0.5, 2.5), (-0.5, 2),
                (-1.5, 3), (-1.5, 2.5), (-1.5, 2), (-1.5, 1.5),
                (-2, 3), (-2, 2.5), (-2, 2), (-2, 1.5), (-2.5, 3), (-2.5, 2.5),
                (-3.1, 3), (-3.1, 2.5), (-3.1, 2), (-3.1, 1.5),
                (-3.5, 3), (-3.5, 2.5), (-3.5, 2), (-3.5, 1.5),
                (-4.0, 3), (-4.0, 2.5), (-4.0, 2), (-4.0, 1.5),
                (-5.5, 3), (-5.5, 2.5), (-5.5, 2), (-5.5, 1.5), (-5.5, 1),
                (-6, 3), (-6, 2.5), (-6, 2), (-6, 1.5), (-6, 1),
                (-6.5, 3), (-6.5, 2.5), (-6.5, 2), (-6.5, 1.5), (-6.5, 1),
                (-6.5, 0.5), (-6.5, 0), (-6.5, -0.5), (-6.5, -1), (-6.5, -1.5),
                (-6, -1), (-6, -1.5), (-5.5, -1), (-5.5, -1.5), (-6.5, -3),
                (-6, -3), (-5.5, -3), (-5, -3),
                (-4.5, -3), (-4, -3), (-4.5, -2.5), (-4, -2.5),
                (-4.5, -3.2), (-4, -3.2), (-3.5, -3.2), (-3, -3.2),
                (-2.5, -3.2), (-2, -3.2), (-1.5, -3.2), (-1, -3.2),
                (-0.5, -3.2), (0.0, -3.2), (0.5, -3.2), (1, -3.2),
                (1.5, -3.2), (2.0, -3.2), (1.5, -2.5), (2.0, -2.5),
                (3.5, -3.2), (4, -3.2), (3.5, -2.5), (4.0, -2.5),
                (4.5, -3.2), (5, -3.2), (4.5, -2.5), (5, -2.5),
                (5.5, -2.5), (5.5, -3.2),
                (4, 0), (3.5, 0), (3, 0), (3, -0.5), (3, -1),
                (-1.5, 1.5), (1.5, 1.5), (-0.2, 1.5), (-1, 1.5), (-5.0, 1.5)
            ]
            return random.choice(points)
        # ── Stages 7 and 8 ─────────────────────────────────────────────────────
        # Stage 7: 5×5 m arena (same outer walls as stages 1–4), six inner walls.
        # Wall positions (from obstacles_stage7/model.sdf, confirmed in stage_map.py):
        #   w1 horizontal at (-1.0, -1.5)  → blocks x∈[-1.5,-0.5], y≈-1.5
        #   w2 vertical   at ( 0.5, -1.0)  → blocks x≈ 0.5, y∈[-1.5,-0.5]
        #   w3 vertical   at ( 1.5, -0.5)  → blocks x≈ 1.5, y∈[-1.0, 0.0]
        #   w4 vertical   at (-1.5,  1.0)  → blocks x≈-1.5, y∈[ 0.5, 1.5]
        #   w5 horizontal at (-0.5,  0.5)  → blocks x∈[-1.0, 0.0], y≈ 0.5
        #   w6 vertical   at ( 0.0,  1.5)  → blocks x≈ 0.0, y∈[ 1.0, 2.0]
        # Points are placed in open areas, all ≥ 0.8 m from origin (MIN_GOAL_DIST).
        elif self.stage == 7:
            points = [
                # bottom strip (y = -2.0)
                (-2.0, -2.0), (-1.0, -2.0), (0.0, -2.0), (1.0, -2.0), (2.0, -2.0),
                # y = -1.5: skip x ∈ [-1.5, -0.5] (w1) and x ≈ 0.5 (w2)
                (-2.0, -1.5), (0.0, -1.5), (1.0, -1.5), (2.0, -1.5),
                # y = -1.0: skip x ≈ 0.5 (w2)
                (-2.0, -1.0), (-1.0, -1.0), (1.0, -1.0), (2.0, -1.0),
                # y = -0.5: skip x ≈ 0.5 (w2) and x ≈ 1.5 (w3)
                (-2.0, -0.5), (-1.0, -0.5), (1.0, -0.5), (2.0, -0.5),
                # y = 0.0: skip x ≈ 1.5 (w3)
                (-2.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (2.0, 0.0),
                # y = 0.5: skip x ∈ [-1.0, 0.0] (w5) and x ≈ 1.5 (w3)
                (-2.0, 0.5), (1.0, 0.5), (2.0, 0.5),
                # y = 1.0: skip x ≈ -1.5 (w4)
                (-2.0, 1.0), (-0.5, 1.0), (1.0, 1.0), (2.0, 1.0),
                # y = 1.5: skip x ≈ -1.5 (w4) and x ≈ 0.0 (w6)
                (-2.0, 1.5), (-0.5, 1.5), (0.5, 1.5), (2.0, 1.5),
                # top strip (y = 2.0): skip x ≈ 0.0 (w6)
                (-2.0, 2.0), (-1.0, 2.0), (0.5, 2.0), (1.0, 2.0), (2.0, 2.0),
            ]
            return random.choice(points)
        # Stage 8: 7.5×7.5 m empty arena (outer_walls_stage5; no inner obstacles).
        # Uses the same outer walls as stage 5; sampling range matches stage 5's
        # maximum extent (±3.0 m) leaving a comfortable margin from the ±3.625 m walls.
        elif self.stage == 8:
            return (random.uniform(-3.00, 3.00), random.uniform(-3.00, 3.00))
        else:
            raise ValueError(
                f"Stage {self.stage} is not a recognised training stage. "
                "Supported stages are 1–8. Check --stage and the Gazebo launch file."
            )

    def generate_random_target_position(self):
        # Robot always resets to (0.0, 0.0) via /reset_simulation.
        robot_x, robot_y = 0.0, 0.0

        best_x, best_y, best_dist = 0.0, 0.0, -1.0
        for _ in range(MAX_GOAL_SPAWN_ATTEMPTS):
            x, y = self._sample_target_position()
            dist = math.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2)
            if dist > best_dist:
                best_dist, best_x, best_y = dist, x, y
            if dist >= MIN_GOAL_DIST:
                self.target_x, self.target_y = x, y
                return self.target_x, self.target_y

        # Fallback: farthest candidate seen across all attempts
        self.target_x, self.target_y = best_x, best_y
        return self.target_x, self.target_y

    def handle_spawn_result(self, future, fixed_z):
        if future.result() is not None:
            self.get_logger().info(f"Entity spawned successfully at coordinates: x={self.target_x}, y={self.target_y}, z={fixed_z}.")
        else:
            self.get_logger().error("Failed to spawn entity.")

    def handle_despawn_result(self, future):
        if future.result() is not None and future.result().success:
            self.get_logger().info("Mark deleted successfully.")
        else:
            self.get_logger().info("No mark to delete or deletion failed.")

    def publish_action(self, action):
        linear_vel = np.abs(float(action[0])) * 0.1
        angular_vel = float(action[1]) * 2 * 0.1
        self.publish_vel(linear_vel, angular_vel)


class Turtle(gym.Env):
    def __init__(self, stage, max_steps, lidar, run_name='baseline', mode='train',
                 odometry_mode='none', device='cpu', resource_logging=False,
                 reward_mode='default', reward_progress_scale=1.0,
                 reward_step_penalty=0.01, reward_turn_penalty=0.01,
                 reward_near_obstacle_scale=0.1, reward_near_obstacle_sigma=0.25,
                 pbrs_scale=1.0, pbrs_distance_weight=1.0, pbrs_angle_weight=0.2,
                 pbrs_distance_scale=5.0, pbrs_gamma=0.997,
                 csv_dir='./csv_logs', plots_dir='./path_plots'):
        super(Turtle, self).__init__()
        self._env = Env(stage, max_steps, lidar, run_name, mode, odometry_mode,
                        device, resource_logging,
                        reward_mode, reward_progress_scale,
                        reward_step_penalty, reward_turn_penalty,
                        reward_near_obstacle_scale, reward_near_obstacle_sigma,
                        pbrs_scale, pbrs_distance_weight, pbrs_angle_weight,
                        pbrs_distance_scale, pbrs_gamma,
                        csv_dir, plots_dir)

        self.observation_space = spaces.Dict({
            'sensor_readings': spaces.Box(low=np.zeros(lidar, dtype=np.float32),
                                    high=np.ones(lidar, dtype=np.float32),
                                    shape=(lidar,),
                                    dtype=np.float32),
            'target': spaces.Box(low=np.zeros(2, dtype=np.float32),
                                    high=np.ones(2, dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32),
            'velocity': spaces.Box(low=np.zeros(2, dtype=np.float32),
                                    high=np.ones(2, dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32),
        })

        self.action_space = spaces.Box(low=np.array([0, -1.]), high=np.array([1., 1.]), dtype=np.float32)

        _odom_shapes = {'twist': (2,), 'delta': (3,), 'full': (5,), 'full_imu': (7,)}
        if odometry_mode in _odom_shapes:
            odom_shape = _odom_shapes[odometry_mode]
            self.observation_space.spaces['odometry'] = spaces.Box(
                low=-np.ones(odom_shape, dtype=np.float32),
                high= np.ones(odom_shape, dtype=np.float32),
                shape=odom_shape,
                dtype=np.float32,
            )

    def step(self, action):
        reward, done, obs = self._env.step(action)

        result = {
            'sensor_readings': obs[:self._env.lidar],
            'target': obs[self._env.lidar:-2],
            'velocity': obs[self._env.lidar + 2:],
            "image": np.zeros((4, 4, 3), dtype=np.int8),
            'is_first': False,
            'is_last': False,
            'is_terminal': done
        }
        if self._env.odometry_mode != 'none':
            result['odometry'] = np.array(self._env._odom_features, dtype=np.float32)

        if done:
            result['log_episode_number'] = float(getattr(self._env, 'log_episode_number', 0))
            result['log_success'] = float(getattr(self._env, 'log_success', 0))
            result['log_steps_to_goal'] = float(getattr(self._env, 'log_steps_to_goal', -1))
            result['log_path_directness'] = float(getattr(self._env, 'log_path_directness', 0))
            result['log_min_obstacle_dist'] = float(getattr(self._env, 'log_min_obstacle_dist', 0))
            result['log_near_collisions'] = float(getattr(self._env, 'log_near_collisions', 0))
            result['log_success_rate'] = float(getattr(self._env, 'log_success_rate', 0))
            result['log_collision_rate'] = float(getattr(self._env, 'log_collision_rate', 0))

        info = {'discount': 0.99}
        if done:
            info['outcome'] = getattr(self._env, 'last_outcome', None)
        return result, reward, done, info

    def reset(self):
        obs = self._env.reset()
        result = {
            'sensor_readings': obs[:self._env.lidar],
            'target': obs[self._env.lidar:-2],
            'velocity': obs[self._env.lidar + 2:],
            'image': np.zeros((4, 4, 3)),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }
        if self._env.odometry_mode != 'none':
            result['odometry'] = np.array(self._env._odom_features, dtype=np.float32)
        return result

    def close(self):
        self._env.destroy_node()