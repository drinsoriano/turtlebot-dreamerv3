# Implementation Audit

An evidence-based audit of what is **actually implemented** in the current
DreamerV3-based autonomous mobile robot navigation project. Every finding is
grounded in repository source code, configuration, or scripts and carries a
confidence label:

- **implemented** — direct, unambiguous code evidence.
- **inferred from implementation** — reasoned from code but not stated explicitly.
- **not found in the current implementation** — no supporting code/config located.

Audit basis: static, read-only inspection only. No training run was launched,
no simulator was started, and no git operations were performed.
Active branch: `reward-shaping-optuna`. All paths are relative to the repository
root; line numbers refer to files as inspected.

**Post-audit update (2026-06-09):** the stage-coverage discrepancy flagged in
§16 was resolved — goal-sampling support for stages 7 and 8 was added to
`turtle.py` after a full Phase 1 verification that every other prerequisite
(launch files, world files, outer-wall SDF, obstacle SDF, A* arena geometry,
robot reset, collision/timeout logic) was already in place. Effective supported
stages are now **1–8**. See §16 for the full resolution record.

---

## 1. Repository Overview

| Area | Path | Relationship to current work |
|---|---|---|
| **DreamerV3 project (in scope)** | `dreamerv3-torch/` | The active model-based RL agent, environment wrapper, training/eval, logging, dashboard, and tuning. |
| Gazebo simulation assets | `turtlebot3_gazebo/` | Launch files, worlds, and models for the TurtleBot3 stages. Assets, not algorithm code. |
| Saved checkpoints / plots | `best_models/`, `plots/` | Output artefacts. |
| **Legacy (excluded)** | `model_free/ddpg/`, `model_free/sac/`, `model_free/td3/`, `model_free/util/` | Model-free implementations (DDPG, SAC, TD3). **Not** called by the DreamerV3 workflow. |
| **Legacy (excluded)** | root `train.py`, `tools.py`, `test.py`, `plt.py`, `save_to_best.py`, and `turtle_env/` | Old model-free harness/scripts at the repository root. Distinct from `dreamerv3-torch/tools.py`. |
| Build/log artefacts | `build/`, `install/`, `log/` | ROS2 build and log output. |

**Likely main project area: `dreamerv3-torch/`** — confidence: **implemented**.
A repository-wide search found no reference to TD3/DDPG/SAC inside
`dreamerv3-torch/`, confirming the DreamerV3 pipeline does not call the legacy
algorithms. The Gazebo world directories named `turtlebot3_dqn_stageN` are
**arena/world names only** — no DQN algorithm code is used (**inferred from
implementation**).

---

## 2. DreamerV3 Related Files Found

| File Path | Purpose | Evidence Found | Confidence |
|---|---|---|---|
| `dreamerv3-torch/dreamer.py` | Training/eval entry point; `Dreamer` agent; `make_env`; main loop; config/CLI merge. | `class Dreamer` (l.29), `make_env` (l.147), `main` loop (l.275–316), argparse auto-register (l.345–348). | implemented |
| `dreamerv3-torch/configs.yaml` | `defaults` + `turtle` config blocks (model, training, behavior, reward, env). | Whole file; `turtle:` block (l.107–122). | implemented |
| `dreamerv3-torch/envs/turtle.py` | ROS2 `Env(Node)` + `Turtle(gym.Env)`: obs/action spaces, reward, termination, goal sampling, CSV logging. | `class Env(Node)` (l.55), `class Turtle(gym.Env)` (l.810). | implemented |
| `dreamerv3-torch/models.py` | `WorldModel` (encoder, RSSM, decoder/reward/cont heads) and `ImagBehavior` (actor/critic over imagined rollouts). | `class WorldModel` (l.29), `class ImagBehavior` (l.213), `_imagine` (l.344), `_compute_target` (l.364). | implemented |
| `dreamerv3-torch/networks.py` | Neural blocks: `RSSM`, `MultiEncoder/MultiDecoder`, `MLP`, `GRUCell`. | `class RSSM` (l.13), `class MultiEncoder` (l.293), `class GRUCell` (l.742). | implemented |
| `dreamerv3-torch/tools.py` | `simulate()` driver, disk-backed replay, `Logger` (white-box CSV). | `class Logger` (l.59), `def simulate` (l.187), `sample_episodes`/`from_generator`/`load_episodes`. | implemented |
| `dreamerv3-torch/envs/stage_map.py` | Stage arena geometry + A* planner for the path-efficiency metric. | `STAGE_ARENAS` (l.29), `astar_plan` (l.264), `get_grid` (l.381), `RESOLUTION=0.05` (l.21). | implemented |
| `dreamerv3-torch/envs/path_viz.py` | Per-episode overhead path plots. | Imported and called in `turtle.py` `_write_planning_csv` (`save_episode_plot`, l.516). | implemented |
| `dreamerv3-torch/envs/resource_logger.py` | Per-episode CPU/RAM/GPU resource logging. | `class ResourceLogger` (l.47), `log_episode` (l.101), `session_id` (l.63). | implemented |
| `dreamerv3-torch/envs/wrappers.py` | Gym wrappers; only `UUID` is applied for this task. | `class UUID` (l.108) applied in `dreamer.py` (l.172). | implemented |
| `dreamerv3-torch/dashboard/app.py` | Streamlit monitoring dashboard over the CSV logs. | `page_title="DreamerV3 Dashboard"` (l.17), `st.tabs([...])` (l.819). | implemented |
| `dreamerv3-torch/tune_reward.py` | Optuna Bayesian-optimisation of shaped-reward weights (wraps `dreamer.py`). | `TPESampler(constraints_func=...)`, `create_study`, subprocess launch of `dreamer.py`. | implemented |
| `dreamerv3-torch/exploration.py` | Exploration behaviors (`Random`, `Plan2Explore`). Not active by default. | `class Random` (l.10), `class Plan2Explore` (l.40); default `expl_behavior: greedy`. | implemented but inactive by default |
| `dreamerv3-torch/parallel.py` | `Parallel`/`Damy` env drivers. Single env used for `turtle`. | Referenced in `dreamer.py` (l.19, 210–215); `envs: 1`. | implemented |
| `dreamerv3-torch/verify_gpu.py` | Standalone GPU/CUDA sanity check. | Separate utility script. | implemented (utility) |

---

## 3. Training Entry Point

**Implemented.** Training is launched via `dreamerv3-torch/dreamer.py`. The
canonical command pattern (from `dreamer.py` and `configs.yaml`):

```
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/<run_name> \
  --stage <N> --lidar <360|10> --odometry_mode <none|twist|delta|full> \
  --seed <S> --device <cpu|cuda> [--steps <int>] [--eval_episode_num <int>] \
  [--reward_mode <default|shaped>] [--reward_* <float> ...]
```

Evidence: `if __name__ == "__main__"` block (`dreamer.py` l.325–349) merges the
`defaults` and `turtle` config blocks and **auto-registers every merged key as a
typed `--key` CLI argument** (l.345–348), so any config value is overridable from
the command line. `main()` (l.176) builds the train/eval envs, prefills, then runs
the alternating train/eval loop.

---

## 4. Configuration System

**Implemented.** Configuration lives in `dreamerv3-torch/configs.yaml`, with a
`defaults` block and a `turtle` override block; CLI flags override both.

Parameters found that are relevant to the requested list:

| Parameter | Found | Default (defaults / turtle) | Location |
|---|---|---|---|
| `stage` | yes | `1` | configs.yaml l.21 |
| `lidar` | yes | `360` | l.22 / l.119 |
| `odometry_mode` | yes | `none` | l.23 / l.120 |
| `seed` | yes | `0` | l.8 |
| `steps` | yes | `300000` / `600000` | l.10 / l.108 |
| `prefill` | yes | `500` | l.39 |
| `eval_every` | yes | `5000` | l.12 |
| `eval_episode_num` | yes | `100` | l.13 |
| `time_limit` | yes | `250` | l.38 |
| `reward_mode` | yes | `default` | l.29 |
| reward knobs | yes | `reward_progress_scale`, `reward_step_penalty`, `reward_turn_penalty`, `reward_near_obstacle_scale`, `reward_near_obstacle_sigma` | l.30–34 |
| logging | yes | `resource_logging: true`, `log_every`, `csv_dir: ./csv_logs`, `log_videos` | l.24, 14, 26, 25 |
| `device` | yes | `cpu` (CLI override to `cuda`) | l.16 / l.117 |

Additional blocks present (**implemented**): world-model/RSSM (`dyn_deter: 256`,
`dyn_stoch: 16`, `dyn_discrete: 16`, `dyn_hidden: 256`; l.43–49), `encoder`/`decoder`
(MLP-only for `turtle`: `mlp_keys: '.*'`, `cnn_keys: '$^'`; l.115–116), `actor`
(normal dist; l.59–60), `critic` (`symlog_disc`, slow target; l.61–62),
`reward_head`/`cont_head`, training (`batch_size: 16`, `batch_length: 64`,
`train_ratio: 512`, `pretrain: 100`, `model_lr: 1e-4`; l.76–84), and behavior
(`discount: 0.997`, `imag_horizon: 15`, `discount_lambda: 0.95`; l.87–89).

---

## 5. Environment and Simulation Interface

**Implemented.** The project connects to a ROS2 / Gazebo simulation of a
**TurtleBot3 burger** via `dreamerv3-torch/envs/turtle.py`:

- `Env(Node)` is a `rclpy` ROS2 node (`turtle.py` l.55).
- **Publishes:** `/cmd_vel` (`geometry_msgs/Twist`) — l.64, `publish_vel` l.377.
- **Subscribes:** `/scan` (`LaserScan`) and `/odom` (`Odometry`) — l.65–66.
- **Service clients:** `/reset_simulation`, `/spawn_entity`, `/delete_entity`,
  `/pause_physics`, `/unpause_physics`, and `/demo/get_entity_state` /
  `/demo/set_entity_state` — l.67–73.
- `Turtle(gym.Env)` (l.810) wraps `Env` and exposes the Gym `observation_space`
  and `action_space`.
- `make_env` (`dreamer.py` l.147) constructs `Turtle(...)` and applies **only**
  the `UUID` wrapper (l.172). The robot resets to the origin `(0,0)` via
  `/reset_simulation`, and goals are spawned as cylinder marker entities
  (`generate_target_sdf`, l.34).

The raw `/odom` pose is used to compute relative-goal features and path metrics;
it is **not** passed directly to the policy (**inferred from implementation**,
`get_state` l.259–268, 296).

---

## 6. Observation Space

**Implemented.** Extracted from `Turtle.observation_space` (`turtle.py`
l.825–850) and `Env.get_state` (l.253–315). The observation is a **`gym.spaces.Dict`**:

| Key | Shape | Contents | Source |
|---|---|---|---|
| `sensor_readings` | `(lidar,)`, default `(360,)` | LiDAR ranges, sub-sampled to `lidar` beams; `inf` clamped to `LIDAR_MAX_RANGE = 3.5` (l.273). | l.826–829 |
| `target` | `(2,)` | `[distance_to_target, angle_to_target]` (bearing in robot frame, wrapped to ±π). | l.830–833; computed l.265–268 |
| `velocity` | `(2,)` | `[linear_vel_cmd, angular_vel_cmd]` — the **previous commanded** action. | l.834–837; l.296 |
| `odometry` | `(2,)` / `(3,)` / `(5,)` | **Optional**, present only when `odometry_mode != none`. | l.842–850 |

The full internal state vector is `lidar + [distance, angle, lin_vel, ang_vel]`
and is squashed with `tanh` before use (l.296–297); `Turtle.step` slices it back
into the dict keys (l.855–863).

A non-encoded **`image`** placeholder (`np.zeros((4,4,3))`) plus boolean flags
(`is_first`, `is_last`, `is_terminal`) are added to the returned observation
(l.859–862) but `image` is **not** part of `observation_space` and is not encoded
(**inferred from implementation**). No camera/CNN perception is used for control.

---

## 7. Action Space and Robot Command Mapping

**Implemented.** Action space (`turtle.py` l.840):

```
action_space = Box(low=[0, -1], high=[1, 1], shape=(2,), dtype=float32)
```

- `action[0]` ∈ `[0, 1]` — forward throttle (magnitude used).
- `action[1]` ∈ `[-1, 1]` — steering / turn command.

Mapping to robot motion (`publish_action`, l.804–807):

```
linear_vel  = abs(action[0]) * 0.1          # up to 0.1 m/s
angular_vel = action[1] * 2 * 0.1           # up to ±0.2 rad/s
publish Twist(linear.x=linear_vel, angular.z=angular_vel) -> /cmd_vel
```

The same `action[0]`/`action[1]` are fed back as the next observation's
`velocity` key. (Constants `num_states=14`, `action_upper_bound=.25`,
`action_lower_bound=.25` exist at l.109–112 but do not size the runtime obs/action
spaces — **inferred from implementation**: legacy/unused.)

---

## 8. Reward Function and Optional Reward Modes

Computed in `get_reward_and_done` (`turtle.py` l.533–638).

**Default reward (`reward_mode='default'`) — implemented.** Sparse terminal reward:

| Event | Condition | Reward |
|---|---|---|
| Success | `distance < REACH_TRESHOLD` (`0.4 m`, l.24) | `+100` (l.542) |
| Collision | `min(lidar) < COLISION_TRESHOLD` (`0.13 m`, l.26) | `-10` (l.564) |
| Timeout | `step_counter >= max_steps - 1` | `-10` (l.586) |
| Otherwise | — | `0` |

**Shaped reward (`reward_mode='shaped'`) — implemented.** Additive on top of the
unchanged terminal reward (l.604–616), gated by `if self.reward_mode == 'shaped'`
(l.607):

| Term | Formula | Weight (default) |
|---|---|---|
| Progress | `scale * (prev_distance - distance)` | `reward_progress_scale=1.0` |
| Step penalty | `- reward_step_penalty` | `0.01` |
| Turn penalty | `- reward_turn_penalty * abs(ang_vel_cmd)` | `0.01` |
| Near-obstacle | `- scale * exp(-d_min/sigma)` when `d_min < 0.2 m` | `scale=0.1`, `sigma=0.25` |

**Other reward modes:**
- Traffic shaping — **not found in the current implementation**.
- A*/planner-guided reward — **not found in the current implementation** (A* is
  used only as a logged metric; see §12). The only branch on `reward_mode` is for
  `'shaped'`; `default` is a no-op shaping path.

Per-episode reward-component sums are written to a reward CSV in both modes
(l.629–636).

---

## 9. Termination Conditions and Episode Horizon

**Implemented.** Three mutually exclusive terminal outcomes in
`get_reward_and_done` (`turtle.py`):

- **Success** — goal within `REACH_TRESHOLD = 0.4 m` (l.539).
- **Collision** — `np.min(lidar) < COLISION_TRESHOLD = 0.13 m` (l.561).
- **Timeout** — `step_counter >= max_steps - 1` (l.583).

**Episode horizon is config-driven** (**implemented**): `max_steps` is the
`time_limit` config value (default `250`, `configs.yaml` l.38), passed as
`max_steps` into `Turtle`/`Env` via `make_env` (`dreamer.py` l.157), and divided
by `action_repeat` in `main` (l.186). It is enforced inside the environment by the
`step_counter` check, **not** by a gym `TimeLimit` wrapper (that wrapper exists in
`wrappers.py` l.7 but is not applied — **inferred from implementation**).

---

## 10. Odometry Ablation or Observation Variants

**Implemented.** Observation variants are controlled by `odometry_mode`. Mode
names and shapes extracted from `turtle.py` (l.275–294 and l.842–850):

| Mode | `odometry` key | Contents |
|---|---|---|
| `none` (default) | absent | Baseline observation, unchanged. |
| `twist` | `(2,)` | `[odom_linear_x, odom_angular_z]` (l.289). |
| `delta` | `(3,)` | `[delta_x_local, delta_y_local, delta_yaw]` in robot frame (l.291). |
| `full` | `(5,)` | twist and delta concatenated (l.293). |

The odometry block is `tanh`-normalised (l.294) and appended as a separate
`odometry` observation key only when the mode is not `none`. Separate
`prev_odom_*` variables track the odometry deltas, distinct from the path-length
variables (l.142–144).

---

## 11. Evaluation and Metrics

**Implemented.** Evaluation runs periodically inside the main loop
(`dreamer.py` l.278–299): every fourth iteration (when `eval_episode_num > 0`) it
runs `eval_episode_num` episodes on a separate eval env with the **deterministic**
actor (`agent(..., training=False)` → `actor.mode()`, `dreamer.py` l.98–100), and
saves the best-mean-return checkpoint as `best.pt`.

Metrics actually computed/logged:

| Metric | Found | Where |
|---|---|---|
| Success / collision (outcome, cumulative + rolling 100/500 rates) | yes | black-box CSV (`turtle.py` `_write_blackbox_csv` l.384) |
| Steps to goal | yes | black-box (`steps_to_goal`) |
| Path directness (`initial_distance / actual_path_length`, capped 1.0) | yes | black-box (l.410) |
| A* path efficiency (`planned / actual`, region + centre) | yes | planning CSV (`_write_planning_csv` l.471) |
| Minimum obstacle distance | yes | black-box (`min_obstacle_dist`) |
| Near-collision count (steps with `min lidar < 0.2 m`) | yes | black-box (l.308–309) |
| Episode return (train & eval) | yes | white-box CSV via `Logger` (`tools.py` l.287, 318) |
| Reward variance | yes | white-box (`tools.py` l.291–294) |
| Model / actor / value losses, KL, prior/posterior entropy | yes | white-box (`tools.py` l.87–91) |
| Reward-component sums | yes | reward CSV (`turtle.py` l.629–636) |
| Resource cost (CPU/RAM/GPU, timing) | yes | resource CSV (`resource_logger.py`) |

---

## 12. A Star or Classical Planner Reference

**Implemented as a metric/reference and visualization — not for control.**

- Planner code: `envs/stage_map.py` — `astar_plan` (l.264), `astar_path_length`
  (l.363), `get_grid` (l.381), `build_grid` (l.200), `STAGE_ARENAS` (l.29),
  `STAGE_OBSTACLES` (l.57), `RESOLUTION = 0.05` (l.21).
- Used **only** by `turtle.py` `_compute_planned_path` (l.441) → `_write_planning_csv`
  (l.471), which logs `planner_path_efficiency` (planned shortest-path length
  divided by actual path length, with a goal-region and a goal-centre variant) and
  triggers the per-episode path plot (`path_viz.save_episode_plot`, l.516).
- A repository search found **no** reference to `astar`/`planner`/`_compute_planned`
  in `dreamer.py`, `models.py`, or `networks.py`.

Conclusion: A* is for **metric/reference + visualization only**; it does **not**
influence navigation control, reward, observations, or the policy.

---

## 13. Bayesian Optimization or Tuning Layer

**Implemented.** `dreamerv3-torch/tune_reward.py` is a standalone Optuna study
that searches the five shaped-reward weights:

- `optuna.samplers.TPESampler(constraints_func=...)` and `optuna.create_study(...
  storage=sqlite:///..., load_if_exists=True)`.
- **Constrained-efficiency objective:** maximise eval `planner_path_efficiency`
  subject to eval `success_rate >= baseline_success - margin`.
- It **wraps `dreamer.py`** as a subprocess with candidate `--reward_*` flags and
  scores each run from the eval CSVs; it does **not** modify the agent,
  observation/action spaces, or A* (A* is read only as the scoring metric).
- A default-reward baseline run fixes the success-rate constraint floor.

This is an **optional research/tuning layer**, separate from a normal `dreamer.py`
training run.

---

## 14. Logging and Data Collection

**Implemented.** Per-run CSVs are written under `csv_dir` (default `./csv_logs/`),
with training and evaluation episodes routed to separate files at
environment-construction time.

**Per-episode metrics (written by the environment, `turtle.py`):**
- `blackbox_{run}.csv` / `blackbox_eval_{run}.csv` — behavioural metrics
  (`_write_blackbox_csv`, header l.185–192).
- `planning_{run}.csv` / `planning_eval_{run}.csv` — A* efficiency
  (`_init_planning_csv` l.422, region + centre columns).
- `reward_{run}.csv` / `reward_eval_{run}.csv` — reward-component sums (l.240–244).
- `resource_{run}.csv` / `resource_eval_{run}.csv` — compute cost
  (`resource_logger.py`).

**Interval / model diagnostics (written by `Logger`, `tools.py`):**
- `whitebox_{run}.csv` — `datetime, step, train_return, reward_variance,
  model_loss, actor_loss, value_loss, kl, prior_ent, post_ent, eval_return,
  eval_success_rate, eval_collision_rate` (l.87–91).

**Visualisation:** per-episode overhead path-plot PNGs (`envs/path_viz.py`),
training vs `_eval` subfolders.

**Dashboard:** `dashboard/app.py` (Streamlit) reads the black-box, white-box,
planning, and resource CSVs and renders tabs
(`overview / blackbox / whitebox / planning / resource`, l.819) plus a sidebar
folder picker for `csv_logs/` and per-experiment subfolders.

> Note (fixed 2026-06-09): the white-box CSV path now honours `csv_dir`
> (`{csv_dir}/whitebox_{run}.csv`, `tools.py` l.82), co-located with the
> black-box/planning/reward/resource CSVs. The `Logger` takes a `csv_dir` argument
> passed from `config.csv_dir` in `dreamer.py`. Files written before the fix remain
> in `./csv_logs/`.

---

## 15. Current Research-Relevant Features (safe to document)

Features confirmed implemented and safe to include in a methodology:

1. **DreamerV3 model-based agent** — MLP encoder, RSSM latent dynamics, decoder /
   reward / continue heads, actor and critic over imagined rollouts (`models.py`,
   `networks.py`, `configs.yaml`).
2. **ROS2/Gazebo TurtleBot3 environment** — `/scan` + `/odom` in, `/cmd_vel` out
   (`turtle.py`).
3. **Dictionary observation space** — `sensor_readings`, `target`, `velocity`
   (+ optional `odometry`), tanh-normalised.
4. **2-D continuous action** mapped to bounded linear/angular velocity.
5. **Sparse default reward** plus an **opt-in additive shaped reward**.
6. **Three terminal conditions** (success/collision/timeout) with a config-driven
   step horizon.
7. **Odometry ablation modes** (`none/twist/delta/full`).
8. **Black-box and white-box metric logging**, plus reward-component and resource
   logging, with train/eval separation.
9. **A\* post-hoc path-efficiency metric** (region + centre variants) and path
   plots — reference/metric only.
10. **Optuna Bayesian-optimisation** of the shaped-reward weights, with a
    constrained-efficiency objective.
11. **Streamlit monitoring dashboard** over the CSV logs.

---

## 16. Not Found or Unclear Features (need confirmation)

- **Stage coverage — resolved (2026-06-09).** The original audit found that
  `_sample_target_position` defined start/goal layouts for **stages 1–6 only**,
  while `STAGE_ARENAS` in `stage_map.py` defined stages 1–8. A Phase 1 inspection
  confirmed that all other prerequisites for stages 7 and 8 were already in place:
  - `turtle_stage7.py` and `turtle_stage8.py` launch files (**implemented**).
  - World SDF files `turtlebot3_dqn_stage7/burger.model` and `turtlebot3_dqn_stage8/burger.model` (**implemented**).
  - Stage 7 outer walls: standard 5×5 m `turtlebot3_dqn_world` model (**implemented**).
  - Stage 8 outer walls: `outer_walls_stage5` (7.5×7.5 m) (**implemented**).
  - Stage 7 obstacles: `obstacles_stage7/model.sdf` — 6 walls at verified positions (**implemented**).
  - Stage 8 obstacles: empty SDF (0 bytes) — matches `STAGE_OBSTACLES[8] = []` (**implemented**).
  - A* arena geometry and obstacle geometry in `stage_map.py` verified against SDF files (**implemented**).
  - Robot reset always at (0, 0); collision/timeout logic is generic — no stage-specific changes needed (**implemented**).
  - Goal sampler was the only missing component.
  
  **Fix applied:** `_sample_target_position` in `turtle.py` extended with an
  `elif self.stage == 7:` block (44 discrete points in open regions of the 5×5 m
  arena, explicitly skipping cells near each of the 6 walls) and an
  `elif self.stage == 8:` block (`random.uniform(-3.00, 3.00)` for both axes,
  matching stage 5's effective range in the same outer walls). An `else` clause
  now raises `ValueError` for any unrecognised stage.
  
  **Effective supported training/evaluation stages: 1–8.** Confidence: **implemented**.
- **Depth-camera / CNN perception** — **not found in the current implementation**
  (encoder is MLP-only; the `image` key is an unused placeholder).
- **A*/planner-guided reward (RS-P)** — **not found in the current
  implementation** (A* is metric-only).
- **Traffic or other reward modes** — **not found** (only `default` and `shaped`).
- **Exploration behaviors** — `Random` and `Plan2Explore` exist (`exploration.py`)
  but the configured `expl_behavior` is `greedy`, so they are **not active by
  default** (**inferred from implementation**). Confirm whether any non-greedy
  exploration is in scope.
- **Unused constants** — `num_states=14`, action-bound `.25` (`turtle.py`
  l.109–112) do not size the runtime spaces; confirm they can be ignored in docs.
- **Planning CSV across branches** — present and active in the current branch's
  `turtle.py`; if methodology must match a different branch, confirm parity.

---

## 17. Recommended Documentation Plan (recommendation only — not created here)

Based solely on these findings, the recommended next documentation steps, in
priority order:

1. **Methodology** — author a Chapter 3-style methodology grounded directly on
   this audit (environment, MDP definition, agent, training/eval protocol,
   metrics). Highest value next step.
2. **Metric definitions** — a dedicated doc defining each black-box, white-box,
   planning, reward-component, and resource metric with its exact formula and
   source (several formulas live only in code, e.g. path directness, A* efficiency,
   reward variance).
3. **Reconcile existing `/docs`** — the repository already contains a prior
   `/docs` set (`README`, `research_process_flow`, `system_architecture`,
   `conceptual_framework`, `observation_and_action_space`,
   `training_and_evaluation_pipeline`, `research_framework_summary`). Cross-check
   them against this audit and correct any drift (the **stage 1–6 vs 1–8** point
   was resolved 2026-06-09; the **white-box `csv_dir`** caveat was also resolved
   2026-06-09 — white-box now honours `csv_dir`).
4. **Process flow / system architecture / conceptual framework / observation-action
   / training-evaluation pipeline** — already drafted in the prior `/docs` set;
   update rather than recreate, once the stage-coverage question is confirmed.

> The **stage-coverage** question (§16) is resolved: effective supported stages
> are 1–8. Methodology can now be written with stages 1–8 as the stated scope.
