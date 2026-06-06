# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

TurtleBot3 autonomous navigation using DreamerV3 (model-based RL). The robot receives LiDAR observations in a Gazebo simulation (ROS2) and learns to reach randomly-spawned goals while avoiding obstacles. Eight training stages increase in obstacle complexity.

Active development spans three branches: `reward-shaping` (current — reward-shaping experiments), `planner-efficiency-metric` (A* metric, dashboard, resource logging), and `odometry-observation` (odometry ablation).

## Project Branches

| Branch | Purpose |
|--------|---------|
| `reward-shaping` | **Current active branch** — reward-shaping experiments for path efficiency across all stages |
| `planner-efficiency-metric` | Upstream baseline — A* metric, path plots, dashboard, resource-cost logging |
| `odometry-observation` | Odometry ablation — none / twist / delta / full modes |
| `baseline-csv-working` | Frozen checkpoint — preserved and pushed |

## Prerequisites

- ROS2 humble with TurtleBot3 simulation packages installed (`/opt/ros/humble/share/turtlebot3_gazebo` must exist)
- The project's `turtlebot3_gazebo/` folder is **not** a ROS2 package and does not need to be built. Launch files are invoked via their full path (see Running a Training Session below).
- Python deps for DreamerV3: `pip install -r dreamerv3-torch/requirements.txt`
- `TURTLEBOT3_MODEL=burger` must be set in environment
- matplotlib ≥ 3.8 installed via pip (user-space) — the system matplotlib on Ubuntu is compiled against NumPy 1.x and fails under NumPy 2.x; see Python Environment below.

## Running a Training Session

**Step 1 — start Gazebo (in a separate terminal):**

The project's `turtlebot3_gazebo/` folder is not a built ROS2 package, so `ros2 launch turtlebot3_gazebo turtle_stage1.py` will fail — ROS2 resolves that package name to the system install at `/opt/ros/humble/share/turtlebot3_gazebo`, which does not contain the custom `turtle_stage*.py` launch files. Launch using the full path instead:

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage1.py
```

Replace `turtle_stage1.py` with `turtle_stage2.py` etc. for other stages. The Gazebo stage number must match the `--stage` argument passed to `dreamer.py`.

Verify the simulation is ready before starting training:
```bash
ros2 service list | grep reset       # expect /reset_simulation
ros2 topic list | grep -E "/odom|/scan|/cmd_vel"  # expect all three
```

**Step 2 — train DreamerV3:**
```bash
cd dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0 \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda
```

See **Example Commands** for smoke test and full training variants.

## Odometry Ablation

`odometry_mode` is a CLI flag controlling what (if any) odometry-derived features are appended as a separate `odometry` observation key:

| Mode | `odometry` shape | Contents |
|------|-----------------|----------|
| `none` (default) | — (key absent) | Corrected baseline — no change to obs space |
| `twist` | (2,) | `[odom_linear_x, odom_angular_z]` from `/odom` twist |
| `delta` | (3,) | `[Δx_local, Δy_local, Δyaw]` in robot frame |
| `full` | (5,) | Both twist and delta combined |

Ablation logdir naming convention: `./logdir/stage{N}_{lidar}_{mode}_seed{S}`

Each mode must use a **fresh logdir** — episode archives (`.npz`) and checkpoints are not compatible across modes.

### Baseline Clarification

`none` mode is a **corrected baseline**, not a clean replica of Paul/Raul's archived baseline. The old codebase hardcoded `turtle.Turtle(4, 300, 360)` in `dreamer.py`, bypassing the configurable odometry/lidar parameters. Paul's archived ~85% success result must not be directly compared to the corrected `none`. The valid comparison is within the corrected ablation: `none vs twist vs delta vs full`.

## A* Planner-Based Path Efficiency

Implemented on `planner-efficiency-metric`. This is a **post-hoc evaluation metric** — it does not change reward, observation space, odometry modes, model architecture, or training logic.

**Metric:**
```
planner_path_efficiency = planned_path_length / actual_path_length
```
Where `planned_path_length` is the A* shortest path from episode start to goal region.

**Separate CSV:** `csv_logs/planning_{run_name}.csv`

Columns: `datetime, stage, episode, outcome, start_x, start_y, target_x, target_y, initial_distance, actual_path_length, planned_path_length, planner_path_efficiency, planner_path_efficiency_raw, planner_status`

**`planner_status` values:**
- `ok` — valid A* path found; numeric columns are populated
- `no_path` — goal region entirely blocked; A* exhausted
- `planner_error` — unexpected exception inside planner
- `unsupported_stage` — stage not in `stage_map.py` geometry tables
- `zero_actual_path` — episode ended before any movement

Numeric fields are **blank (empty string)** for non-`ok` rows — never `-1.0` — so pandas/CSV tools treat them as NaN naturally.

## Reward Shaping (branch: `reward-shaping`)

**Goal:** Improve navigation path efficiency across all stages — reduce looping, wandering, and overshooting behavior without breaking success rate.

**Baseline:** `planner-efficiency-metric` branch. All A* metrics, dashboard, and resource-cost logging are inherited unchanged and remain diagnostic/logging only. The A* `planner_path_efficiency` metric is the primary way to measure improvement.

**Scope of changes in this branch:**
- `dreamerv3-torch/envs/turtle.py` — `get_reward_and_done()` (additive shaping after the **unchanged** terminal block), `step()` (passes the angular command), `init_properties()` / `reset()` / constructors (store knobs, reset per-episode component sums, open the reward CSV).
- `dreamerv3-torch/configs.yaml` — six reward-shaping config knobs (see below).
- `dreamerv3-torch/dreamer.py` — `make_env()` only, to forward the knobs into `Turtle(...)`.

**Do not change** in this branch: observation space, odometry modes, DreamerV3 architecture, A* metric computation, dashboard, resource logging, torch/CUDA setup.

### Reward mode (implemented)

Shaping is **opt-in** via `--reward_mode`. The original reward is preserved exactly:

| `reward_mode` | Behavior |
|---|---|
| `default` (default) | Original sparse reward, **byte-for-byte unchanged**: `+100` goal, `−10` collision, `−10` timeout, `0` otherwise |
| `shaped` | Original terminal rewards **plus** the additive shaping terms below, applied every step |

Shaped reward `= reward_default + progress + step_pen + turn_pen + near_obst`:

| RS | Term | Formula | Config knob (default) |
|---|---|---|---|
| RS-1 | Progress reward | `+scale * (prev_dist − curr_dist)` | `reward_progress_scale` (1.0) |
| RS-2 | Step penalty | `−c` per step | `reward_step_penalty` (0.01) |
| RS-3 | Turning penalty | `−k * |ang_vel_cmd|` | `reward_turn_penalty` (0.01) |
| RS-4 | Near-obstacle penalty | `−k * exp(−d_min / σ)` when `d_min < 0.3 m` | `reward_near_obstacle_scale` (0.1), `reward_near_obstacle_sigma` (0.25) |

Weights are kept **mild by default** — progress is deliberately weak so complex stages can still take temporary detours around obstacles. All knobs are CLI-overridable.

**Component logging:** per-episode component sums are written to `csv_logs/reward_{run_name}.csv` (eval → `reward_eval_{run_name}.csv`) with columns `datetime, reward_mode, stage, episode, outcome, sum_terminal, sum_progress, sum_step_penalty, sum_turn_penalty, sum_near_obstacle, sum_total`. Written in **both** modes; in `default` mode the shaping columns are `0` and `sum_total == sum_terminal`. The blackbox/planning/whitebox/resource CSV schemas are unchanged.

**A* planner-guided reward:** Not approved. Do not implement until explicitly scoped. If added later, label the experiment `RS-P` and isolate in a separate sub-experiment logdir.

### Logdir naming convention for reward-shaping experiments

```
./logdir/stage{N}_360_none_seed{S}_reward_{default|shaped}
```

Example: `stage1_360_none_seed0_reward_shaped` for the shaped run on stage 1, `stage1_360_none_seed0_reward_default` for the no-shaping baseline. Replace `{N}` with the target stage number.

### Measuring improvement

Compare against a no-shaping baseline run on this branch for the same stage (not `planner-efficiency-metric`):
- Primary: `planner_path_efficiency` from `csv_logs/planning_{run_name}.csv`
- Secondary: `path_directness` from `blackbox_{run_name}.csv`
- Guard: `success_rate` and `collision_rate` must not regress

## Path Plots

PNG overhead plots are generated per episode under `dreamerv3-torch/path_plots/{run_name}/`.

**Filename pattern:** `ep{episode:05d}_{outcome}.png` (e.g. `ep00001_timeout.png`, `ep00042_success.png`)

**Generated whenever:** A* returns a valid path and the episode ends (success / collision / timeout). No toggle flag — always on. Disable during long training runs if I/O becomes a bottleneck.

**Each plot shows:**
- Stage arena boundary and physical obstacle outlines (from `stage_map.py` geometry)
- A* reference path (blue)
- A* endpoint with validity annotation (blue if inside goal region, red if outside)
- Goal acceptance circle (dashed red, radius = 0.4 m = `REACH_THRESHOLD`)
- Goal fill region (light red)
- Robot trajectory (orange)
- Start position (green circle)
- Goal centre (red star)

**Circular Gazebo goal marker:** The visual marker spawned in Gazebo was changed from a flat plane (square) to a cylinder matching the goal acceptance radius. This is visual-only — it has no effect on collision detection, LiDAR readings, reward, observation space, or training logic.

**Orientation note (pending fix):** In Gazebo top view, +X points upward. In the current PNG, +X points right, creating a 90° misalignment. The intended display transform is:
```
display_x = -world_y
display_y =  world_x
```
Apply consistently to A* path, robot trajectory, start, goal, obstacles, and arena boundaries. This is visualization only — A* metric values are unaffected.

`path_plots/` is gitignored — never commit PNG files.

## Monitoring (Streamlit Dashboard)

The preferred monitoring tool is the Streamlit dashboard:

```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch/dashboard
streamlit run app.py
```

`live_chart.py` is no longer used.

## Gazebo Launch Notes

- **Preferred launch form (direct path):**
  ```bash
  export TURTLEBOT3_MODEL=burger
  ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage{n}.py
  ```
- `turtlebot3_gazebo/` is **not** a built ROS2 package. Do not use `ros2 launch turtlebot3_gazebo ...` (resolves to system install at `/opt/ros/humble`).
- Launch files resolve local paths via `os.path.dirname(os.path.realpath(__file__))` — **not** `get_package_share_directory('turtlebot3_gazebo')`. Keep `get_package_share_directory('gazebo_ros')` as-is (real system package).
- `GAZEBO_MODEL_PATH` is prepended to the local `models/` directory via `SetEnvironmentVariable` inside each launch file. Note: `GazeboRosPaths.get_paths()` in `gzserver.launch.py` injects the system turtlebot3_gazebo models first, so system SDF files take priority. Modifying system SDF files requires root.
- **REP 145 IMU warning** (`<initial_orientation_as_reference> is unset, using default value of false`) is harmless. DreamerV3 subscribes only to `/scan` and `/odom`, not `/imu`. The warning cannot be silenced without root access to the system SDF at `/opt/ros/humble/share/turtlebot3_gazebo/models/`.

## GPU Setup

**Hardware:** Dell Latitude 5430, i5, 32 GB RAM + RTX 5060 Ti 16 GB via Thunderbolt 4 eGPU.

- `nvidia-smi` detects the RTX 5060 Ti; `torch.cuda.is_available()` = True.
- The RTX 5060 Ti is Blackwell (SM 120) and requires **CUDA 12.8+ / torch 2.11+**. Do not install `torch==2.0.0` or any CUDA 11.x wheel.
- `configs.yaml` defaults to `device: 'cpu'`. Pass `--device cuda` at the CLI to use GPU.
- Do **not** reinstall or downgrade torch unless explicitly approved.
- After any environment change, verify with:
  ```bash
  cd ~/turtlebot-dreamerv3/dreamerv3-torch
  python3 verify_gpu.py
  ```
  Expected: all four checks `[PASS]` — torch imports, CUDA available, GPU detected, CUDA tensor op.

## Python Environment

**NumPy / matplotlib conflict:**
- `opencv-python 4.13.0` on Python 3.9+ hard-requires `numpy>=2` (wheel metadata: `Requires-Dist: numpy>=2; python_version >= "3.9"`). Do **not** downgrade NumPy — it will break cv2.
- The system matplotlib at `/usr/lib/python3/dist-packages/matplotlib/` (version 3.5.1) was compiled against NumPy 1.x. Under NumPy 2.x it raises `AttributeError: _ARRAY_API not found`.
- **Fix:** install a pip-built matplotlib into user-space to shadow the broken system copy:
  ```bash
  pip install --user "matplotlib==3.8.4"
  ```

**After any environment change, verify all of these:**
```bash
python3 -c "
import numpy, matplotlib, cv2, torch
print('numpy    ', numpy.__version__, numpy.__file__)
print('matplotlib', matplotlib.__version__, matplotlib.__file__)
print('cv2      ', cv2.__version__)
print('torch    ', torch.__version__)
print('CUDA     ', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU      ', torch.cuda.get_device_name(0))
"
```
Expected: numpy 2.x, matplotlib from `~/.local/`, CUDA True, RTX 5060 Ti.

## Training Speed

- Gazebo RTF ≈ 4.5–4.7 (with GUI), ≈ 4.99 (headless). Headless gives only marginal gain.
- More impactful levers:
  - `--device cuda` — GPU training
  - `--eval_episode_num N` — fewer eval episodes per checkpoint = less wall-clock pause
  - `--eval_every N` — less frequent evaluation
- **`eval_episode_num` guidance:**
  - `2` — smoke test / fast iteration only
  - `100` — preferred for real training and final result reporting

## Example Commands

**Smoke test (any stage N):**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/gpu_smoke_stage{n}_none_seed0 \
  --stage {n} --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 5000 --eval_episode_num 2
```

**GPU training (any stage N):**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/stage{n}_360_none_seed0 \
  --stage {n} --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100
```

**GPU training with resource-cost logging:**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/stage{n}_360_none_seed0 \
  --stage {n} --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --resource_logging True
```

Replace `{n}` with the stage number (1–8) and `none` with `twist`, `delta`, or `full` as needed.

## Git Rules

- **Never** use `git add .` or `git add -A`.
- **Never commit** generated output:
  - `logdir/` — episode archives and checkpoints
  - `csv_logs/` — training metrics CSVs
  - `path_plots/` — path visualization PNGs
  - `install/` — install artifacts
  - `log/` — ROS2/system log output
  - `*.pt`, `*.npz`, `*.jsonl`, `events.out.tfevents*`
- Always run before committing:
  ```bash
  git status --short
  git diff --stat
  ```
- Use `git add <specific-file>` only.

## Architecture Overview

### Environment path

| Path | Entry | Used by |
|------|-------|---------|
| `dreamerv3-torch/envs/turtle.py` | `Turtle(gym.Env)` wrapping `Env(Node)` | DreamerV3 |

Subscribes to `/scan` (LiDAR) and `/odom` (odometry), publishes to `/cmd_vel`.

### Baseline observation construction (`Env.get_state()`)

```
state = lidar[0..N-1] + [distance_to_target, angle_to_target, lin_vel_cmd, ang_vel_cmd]
state = tanh(state)   # all values normalized to [-1, 1]
```

`distance_to_target` and `angle_to_target` are computed from `/odom` pose (position + yaw) and the spawned goal position — the raw pose is **never** passed to the policy. `lin_vel_cmd` / `ang_vel_cmd` are the **commanded** action values from the previous step, not odometry-measured velocities.

The `observation_space` dict keys (`sensor_readings`, `target`, `velocity`, and optionally `odometry`) are what `MultiEncoder` in `networks.py` uses to build MLP sub-encoders. The `image` key exists in the obs dict but is **not** in `observation_space` and is not encoded (dummy zero tensor for interface compatibility).

### DreamerV3 data flow

```
Env (ROS2) → Turtle (gym.Env) → UUID wrapper
    ↓
tools.simulate() — collects transitions, calls agent(obs, done)
    ↓ (saves .npz episodes to traindir/)
tools.sample_episodes() → tools.from_generator() → dataset batches
    ↓
Dreamer.__call__() → Dreamer._train(batch)
    ├── WorldModel._train()   (models.py) — RSSM + encoder + decoder heads
    └── ImagBehavior._train() (models.py) — actor + critic via imagined rollouts
```

The encoder (`MultiEncoder` in `networks.py`) reads `observation_space` at construction time. Adding a new key to `observation_space` automatically creates a new MLP sub-encoder — no changes to `networks.py` or `models.py` are needed.

Episode replay is **disk-backed** (`.npz` files), not an in-memory ring buffer. `erase_over_episodes()` in `tools.py` enforces `dataset_size`.

### Config system

`dreamerv3-torch/configs.yaml` has a `defaults` block and a `turtle` override block. CLI args (`--key value`) override both. The argparse loop in `dreamer.py` (lines 330-332) auto-registers every key in the merged config as a typed CLI argument.

Key turtle config values: `steps=600000`, `lidar=360`, `batch_size=16`, `batch_length=64`, `dyn_deter=256`, `dyn_stoch=16`, `dyn_discrete=16`, `encoder: {mlp_keys: '.*', cnn_keys: '$^'}` (MLP-only, no CNN).

### CSV logging

Seven CSV files per run, written under `dreamerv3-torch/csv_logs/`. Train and eval episodes are separated — main CSVs contain only training episodes; eval episodes go to parallel eval CSVs:

| CSV file | Written by | Contents |
|---|---|---|
| `blackbox_{run_name}.csv` | Train env | Per-episode train metrics |
| `blackbox_eval_{run_name}.csv` | Eval env | Per-episode eval metrics (same schema) |
| `whitebox_{run_name}.csv` | `Logger.write()` | Step-based algorithm metrics (always train) |
| `planning_{run_name}.csv` | Train env | A* metrics for train episodes |
| `planning_eval_{run_name}.csv` | Eval env | A* metrics for eval episodes |
| `resource_{run_name}.csv` | Train env | Resource cost (if `--resource_logging True`) |
| `resource_eval_{run_name}.csv` | Eval env | Resource cost for eval (same flag) |

**`blackbox_*` columns:** `datetime, odometry_mode, stage, episode, outcome, steps_to_goal, path_directness, min_obstacle_dist, near_collisions, success_rate, collision_rate, rolling_success_rate_100, rolling_collision_rate_100, rolling_success_rate_500, rolling_collision_rate_500`. Episode numbering and rolling-window counters in each CSV are independent — cumulative rates and rolling rates in `blackbox_eval_*` reflect eval-only performance across all checkpoints.

**`whitebox_{run_name}.csv` columns:** `datetime, step, train_return, reward_variance, model_loss, actor_loss, value_loss, kl, prior_ent, post_ent, eval_return, eval_success_rate, eval_collision_rate`

**`planning_*` columns:** see **A* Planner-Based Path Efficiency** above; written on `planner-efficiency-metric` branch only.

**Routing mechanism:** File routing is done at `init_properties()` time based on `mode` — no runtime mode guards inside write methods. Train env opens `blackbox_{run_name}.csv`; eval env opens `blackbox_eval_{run_name}.csv`. The same resume logic (seed counters from existing CSV on restart) applies to both.

**Path plots:** Train episodes → `path_plots/{run_name}/`. Eval episodes → `path_plots/{run_name}_eval/`. Both are gitignored by `path_plots/`. Over a 300k-step run with 100 eval episodes per checkpoint this generates O(thousands) of eval PNGs — disable A* in eval by patching `_write_planning_csv` if I/O becomes a bottleneck.

`run_name` is derived from the final component of `--logdir` in `make_env()`.

## Key Variable Notes

- `self.prev_x` / `self.prev_y` in `Env` — used **exclusively** for path-length tracking (thesis metrics). Do not reuse for odometry deltas.
- `self.prev_odom_x/y/yaw` — separate variables used for odometry delta computation in non-`none` modes.
- `config.time_limit` is divided by `action_repeat` at the start of `main()` before `make_env()` is called.
- The stage number in `make_env()` comes from `config.stage`; the logdir is set via `--logdir`.

## Metric Interpretation

- `path_directness` in `blackbox_{run_name}.csv` is a coarse metric: `initial_distance / actual_path_length`. It does not account for obstacle detours and is not the thesis-grade path efficiency measure.
- **A* planner-based `planner_path_efficiency`** is the stronger final path-quality metric — it compares the robot's actual path against the optimal A* path for each episode's specific start/goal pair.
- If success rate improves but planner path efficiency remains low, treat it as a valid research finding: DreamerV3 may learn goal-reaching behavior without learning shortest-path behavior.

## Resource-Cost Logging

Enabled via `--resource_logging True` (default `false`, zero overhead when off).
Written to `dreamerv3-torch/csv_logs/resource_{run_name}.csv`, one row per episode.
Implementation: `dreamerv3-torch/envs/resource_logger.py`.

**Metric scope policy — primary columns are process-specific (DreamerV3 only):**

| Column | Scope | Notes |
|---|---|---|
| `cpu_percent_process` | Process only | `psutil.Process.cpu_percent()` |
| `ram_used_mb_process` | Process only | `psutil.Process.memory_info().rss` |
| `gpu_memory_used_mb_process` | Process only | NVML `nvmlDeviceGetComputeRunningProcesses()` filtered by PID |
| `cpu_percent_system` | System-wide (context) | includes all desktop apps |
| `ram_percent_system` | System-wide (context) | includes all desktop apps |
| `gpu_util_percent_device` | Device-wide (context) | NVML has no per-process GPU util |
| `gpu_memory_used_mb_device` | Device-wide (context) | total device used |
| `gpu_power_watts` | Device-wide (context) | NVML |
| `gpu_temperature_c` | Device-wide (context) | NVML |

**Perception/sensor columns** (for future cross-mode comparison):
`perception_mode`, `lidar_beams`, `odometry_mode`, `imu_enabled`, `depth_enabled`,
`depth_camera_count`, `depth_resolution`, `sensor_config_id` (e.g. `lidar360_none`)

**Timing columns:** `episode_wall_time_sec`, `total_wall_time_sec`

**`session_id` column:** A timestamp string (`YYYYMMDD_HHMMSS`) generated once per process invocation. After a restart the new session gets a different `session_id` even if `run_name` is the same. Use `df[df.session_id == '20260606_212018']` to isolate a single run. On startup with an existing CSV, a warning is printed to stdout listing the new `session_id`.

**Phase separation:** Train episodes → `resource_{run_name}.csv`. Eval episodes → `resource_eval_{run_name}.csv`. Routing is determined at `ResourceLogger` construction time (different `run_name` passed for eval). No mode guard at call sites.

**GPU sampling caveats:**
- `device` = training compute device (`cpu` or `cuda`) — not a sensor column.
- `gpu_util_percent_device` is an **instantaneous sample** at episode end. It may read 0% if the training step is not executing at that exact moment. Use it for trend analysis across many episodes, not single-episode interpretation.
- `gpu_memory_used_mb_process` and `gpu_power_watts` are more reliable per-episode indicators: process memory persists between steps and power reflects actual compute load over the episode duration.

**Dependencies:** `psutil` (already installed), `pynvml` / `nvidia-ml-py3` (installed). If either is missing, the affected fields are blank — training never crashes.

## Future Work

### Future perception modes
- `lidar` — current mode
- `depth` — depth camera (future)
- `lidar_depth` — combined (future)
- Depth camera and CNN integration are future work, separate from the current odometry ablation. Do not implement until explicitly scoped.


Test commands (need Gazebo on stage 1)

cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_default \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 5000 --eval_episode_num 2 --reward_mode default

python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_shaped \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 5000 --eval_episode_num 2 --reward_mode shaped