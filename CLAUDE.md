# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

TurtleBot3 autonomous navigation using DreamerV3 (model-based RL). The robot receives LiDAR observations in a Gazebo simulation (ROS2) and learns to reach randomly-spawned goals while avoiding obstacles. Six training stages increase in obstacle complexity.

## Prerequisites

- ROS2 humble with TurtleBot3 simulation packages installed (`/opt/ros/humble/share/turtlebot3_gazebo` must exist)
- The project's `turtlebot3_gazebo/` folder is **not** a ROS2 package and does not need to be built. Launch files are invoked via their full path (see Running a Training Session below). To use the `ros2 launch turtlebot3_gazebo ...` short form instead, copy the launch/models/worlds files into a local ROS2 workspace and rebuild with `colcon build` per `turtlebot3_gazebo/README.md`.
- Python deps for DreamerV3: `pip install -r dreamerv3-torch/requirements.txt`
- `TURTLEBOT3_MODEL=burger` must be set in environment

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
  --stage 1 --lidar 360 --odometry_mode none --seed 0
```

## Odometry Ablation

`odometry_mode` is a CLI flag controlling what (if any) odometry-derived features are appended as a separate `odometry` observation key:

| Mode | `odometry` shape | Contents |
|------|-----------------|----------|
| `none` (default) | — (key absent) | Exact baseline — no change to obs space |
| `twist` | (2,) | `[odom_linear_x, odom_angular_z]` from `/odom` twist |
| `delta` | (3,) | `[Δx_local, Δy_local, Δyaw]` in robot frame |
| `full` | (5,) | Both twist and delta combined |

Ablation logdir naming convention: `./logdir/stage{N}_{lidar}_{mode}_seed{S}`

Each mode must use a **fresh logdir** — episode archives (`.npz`) and checkpoints are not compatible across modes.

## Live Chart Monitoring

`dreamerv3-torch/live_chart.py` is a read-only monitoring tool. It does not affect training and can be run in a separate terminal while `dreamer.py` is running.

**Basic usage — derive CSV paths from run name:**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 live_chart.py --run_name stage1_360_none_seed0 --update_every_episodes 100
```

`--run_name stage1_360_none_seed0` reads:
```
./csv_logs/blackbox_stage1_360_none_seed0.csv
./csv_logs/whitebox_stage1_360_none_seed0.csv
```

The `run_name` must match the final component of the `--logdir` used by `dreamer.py`. If training uses `--logdir ./logdir/stage1_360_full_seed0`, then live_chart.py must use `--run_name stage1_360_full_seed0`.

**Odometry ablation modes:**
```bash
python3 live_chart.py --run_name stage1_360_none_seed0  --update_every_episodes 100
python3 live_chart.py --run_name stage1_360_twist_seed0 --update_every_episodes 100
python3 live_chart.py --run_name stage1_360_delta_seed0 --update_every_episodes 100
python3 live_chart.py --run_name stage1_360_full_seed0  --update_every_episodes 100
```

**Explicit CSV paths (when run_name is not convenient):**
```bash
python3 live_chart.py \
  --blackbox ./csv_logs/blackbox_stage1_360_none_seed0.csv \
  --whitebox ./csv_logs/whitebox_stage1_360_none_seed0.csv \
  --update_every_episodes 100
```

**`--update_every_episodes N`** — the chart redraws only when the episode count crosses a new N-episode boundary (100, 200, 300, …). The first draw happens immediately regardless of episode count. Omitting the flag defaults to redrawing every polling interval (every 10 s).

## Architecture Overview

### Two separate environment paths

| Path | Entry | Used by |
|------|-------|---------|
| `dreamerv3-torch/envs/turtle.py` | `Turtle(gym.Env)` wrapping `Env(Node)` | DreamerV3 |
| `turtle_env/turtle.py` | Same structure, no CSV logging | SAC / DDPG / TD3 |

Both subscribe to `/scan` (LiDAR) and `/odom` (odometry) and publish to `/cmd_vel`.

### Baseline observation construction (`Env.get_state()`)

```
state = lidar[0..N-1] + [distance_to_target, angle_to_target, lin_vel_cmd, ang_vel_cmd]
state = tanh(state)   # all values normalized to [-1, 1]
```

`distance_to_target` and `angle_to_target` are computed from `/odom` pose (position + yaw) and the spawned goal position — the raw pose is **never** passed to the policy. `lin_vel_cmd` / `ang_vel_cmd` are the **commanded** action values from the previous step, not odometry-measured velocities.

The `observation_space` dict keys (`sensor_readings`, `target`, `velocity`, and optionally `odometry`) are what `MultiEncoder` in `networks.py` uses to build MLP sub-encoders. The `image` key exists in the obs dict but is **not** in `observation_space` and is not encoded (it is a dummy zero tensor kept for interface compatibility).

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

Two CSV files per run, written under `dreamerv3-torch/csv_logs/`:

- `blackbox_{run_name}.csv` — per-episode metrics written by `Env._write_blackbox_csv()` in `turtle.py`; columns: `datetime, odometry_mode, stage, episode, outcome, steps_to_goal, path_efficiency, min_obstacle_dist, near_collisions, success_rate, collision_rate, rolling_success_rate_100, rolling_collision_rate_100, rolling_success_rate_500, rolling_collision_rate_500`. On startup the existing CSV is read to resume episode numbering and seed the rolling window — prevents artificial drops when appending after a restart. Old CSVs (created before the rolling columns were added) will have the 4 rolling values appended as unlabelled columns on new rows.
- `whitebox_{run_name}.csv` — per-training-step algorithm metrics written by `Logger.write()` in `tools.py`; columns: `datetime, step, train_return, reward_variance, model_loss, actor_loss, value_loss, kl, prior_ent, post_ent, eval_return, eval_success_rate, eval_collision_rate`

`run_name` is derived from the final component of `--logdir` in `make_env()`.


## Key Variable Notes

- `self.prev_x` / `self.prev_y` in `Env` — used **exclusively** for path-length tracking (thesis metrics). Do not reuse for odometry deltas.
- `self.prev_odom_x/y/yaw` — separate variables used for odometry delta computation in non-`none` modes.
- `config.time_limit` is divided by `action_repeat` at the start of `main()` before `make_env()` is called.
- The stage number in `make_env()` comes from `config.stage`; the logdir is set via `--logdir`.

## Current Development Status

Current working branch: `odometry-observation`.

Baseline branch: `baseline-csv-working` is preserved and pushed.

Implemented:
- Configurable `odometry_mode`
- Odometry observation key for `twist`, `delta`, and `full`
- `none` mode preserves baseline observation shape
- `dreamer.py` passes config values into `Turtle`

Pending verification:
- Launch correct custom Gazebo stage
- Run smoke test for `odometry_mode none`
- Run smoke test for `odometry_mode twist`
- Verify blackbox CSV logging continuity after restart