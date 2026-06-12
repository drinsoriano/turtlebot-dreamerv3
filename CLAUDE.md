# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

TurtleBot3 autonomous navigation using DreamerV3 (model-based RL). The robot receives LiDAR observations in a Gazebo simulation (ROS2) and learns to reach randomly-spawned goals while avoiding obstacles. Eight training stages increase in obstacle complexity.

**Stage coverage: 1–8 (fully supported).** The goal sampler `_sample_target_position` in `envs/turtle.py` defines start/goal layouts for stages 1–8 (stages 7 and 8 added 2026-06-09; unknown stage → `ValueError`), and the A* arena geometry (`STAGE_ARENAS` in `envs/stage_map.py`) covers 1–8. Each stage needs its own Gazebo launch file (`turtle_stage{N}.py`) and a matching `--stage N`. A thesis-ready methodology and supporting docs live under `docs/` (`methodology.md`, `implementation_audit.md`, `whitebox_data_validation.md`, etc.).

Active development spans several branches: `imu-observation` (current — IMU `full_imu` mode), `reward-shaping-optuna` (BO tuning checkpoint), `depth-perception` (depth camera integration), `planner-efficiency-metric` (A* metric, dashboard, resource logging), and `odometry-observation` (odometry ablation).

## Project Branches

| Branch | Purpose |
|--------|---------|
| `imu-observation` | **Current active branch** — IMU integration via the `full_imu` odometry mode (`full` + 2D linear acceleration from `/imu`); branched off `reward-shaping-optuna` |
| `reward-shaping-optuna` | BO tuning checkpoint — Optuna reward-weight search; clean checkpoint that IMU and depth branch off |
| `depth-perception` | Depth camera integration — burger SDF, ROS2 `/camera/depth/image_raw`, obs wiring, CNN encoder |
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
ros2 topic list | grep -E "/odom|/scan|/cmd_vel|/imu"  # expect all four (/imu only needed for odometry_mode=full_imu)
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
| `full_imu` | (7,) | `full` (5) **+** `[accel_x, accel_y]` — 2D robot-frame linear acceleration from `/imu` (branch `imu-observation`) |

Ablation logdir naming convention: `./logdir/stage{N}_{lidar}_{mode}_seed{S}`

Each mode must use a **fresh logdir** — episode archives (`.npz`) and checkpoints are not compatible across modes.

**`full_imu` IMU source (no Gazebo change needed):** The burger SDF already runs the IMU plugin (`libgazebo_ros_imu_sensor.so`) publishing `sensor_msgs/Imu` to `/imu` at 200 Hz — confirmed in **both** the local SDF and the system install at `/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf` (the system SDF is the one that loads; see Gazebo Launch Notes). The REP 145 IMU warning is that same plugin at runtime. So `/imu` is available on **all stages 1–8** with no SDF/launch edit. Only `linear_acceleration.x/y` are used; gyro (`angular_velocity.*`) and `accel_z` are intentionally excluded (near-zero information on a flat 2D arena). The `/imu` subscription is created **only** when `odometry_mode == 'full_imu'`, so `none/twist/delta/full` stay zero-overhead and behaviorally unchanged. The resource CSV records `imu_enabled=True` and `sensor_config_id=lidar{N}_full_imu`.

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

**Robot footprint (obstacle inflation) — A\* is not a point robot.** `build_grid` in `stage_map.py` inflates every obstacle **and** the outer walls by `ROBOT_RADIUS = 0.15 m` (= TurtleBot3 Burger footprint ≈ `0.105 m` + `0.045 m` safety margin) at `RESOLUTION = 0.05 m` per cell — the standard configuration-space (Minkowski) approach. So A\* keeps the robot **centre ≥ 0.15 m** from any obstacle and routes through a gap **only if it is wider than ~0.30 m** (2 × radius). Because `0.15 m > ` the real ~`0.105 m` radius, **A\* is more conservative than the physical robot**: any path A\* finds is physically followable (with ~0.045 m clearance), and A\* will **never** squeeze through a gap the robot cannot. In tight layouts (e.g. narrow stage-4 corridors) a passage between ~0.21 m and ~0.30 m is fittable by the real robot but declined by A\* → that episode logs `planner_status = no_path` with blank efficiency (honest missing data, **not** a wrong value, and never an over-optimistic one). To make A\* match the robot more exactly, lower `ROBOT_RADIUS` toward `0.105 m`; the conservative default is preferred so the planned path is guaranteed feasible.

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

### Reward-Weight Tuning (Bayesian Optimization)

`dreamerv3-torch/tune_reward.py` searches the five `shaped`-mode weights with Optuna. It is a **standalone orchestration script** — it launches `dreamer.py` as a subprocess with candidate `--reward_*` flags and scores each run from the eval CSVs. It does **not** modify reward/observation/odometry/architecture/A\*/dashboard/torch. **A\* is used only as the scoring metric, never fed into training.**

- **Objective (constrained efficiency):** maximize eval `planner_path_efficiency` subject to eval `success_rate ≥ baseline_success − margin` (default margin 5 pts). Constraint handled via `TPESampler(constraints_func=...)`.
- **Fidelity:** short proxy budget per trial (`--steps 80000 --eval_episode_num 20`), then validate the winner at full budget. Trials run **sequentially** (one Gazebo, one GPU). At the default `eval_every = 20000`, an **80k** trial yields **5 evals → ~100 eval episodes**, and `SCORE_WINDOW = 60` scores the **last 3 evals** (the 40k/60k/80k checkpoints), excluding the untrained `ctr=0` eval. **Use `--steps ≥80000` for BO:** a 40k trial has only 3 evals (60 rows), so the 60-row window would reach the untrained `ctr=0` eval and pollute the score (for short trials, lower `SCORE_WINDOW`). `tune_reward.py` does not expose `--eval_every`, so trials use the config default. See [docs/evaluation_loop.md](docs/evaluation_loop.md).
- **Persistence:** study saved to `tune_reward_stage{N}.db` (sqlite, gitignored) with `load_if_exists=True` — a crash/reboot resumes the study.

**`tune_reward.py` does not replace `dreamer.py` — it wraps it.** Each trial builds and runs a normal `dreamer.py ... --reward_mode shaped` command with the five `--reward_*` flags chosen by Optuna. A plain `dreamer.py` run (no weight flags) is still the way to do smoke tests, the baseline, and final validation.

#### How to run the BO search

Needs **two terminals**.

**Terminal 1 — start Gazebo once** (stays up for the entire study; the stage is fixed):
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage1.py
```

**Terminal 2 — run the tuner:**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

# (optional) preview the exact dreamer.py commands — no training, no Gazebo needed
python3 tune_reward.py --stage 1 --n-trials 2 --dry-run

# 1) default-reward baseline → sets the success-rate constraint floor
python3 tune_reward.py --stage 1 --run-baseline --steps 80000 --eval-episode-num 20

# 2) the search: 30 trials, each an 80k-step run with Optuna-chosen weights
#    (auto-runs a baseline first if --baseline-success is not given)
python3 tune_reward.py --stage 1 --n-trials 30 --steps 80000 --eval-episode-num 20
```

> **Why `--steps 80000` (not 40k):** with `eval_every = 20000` and `SCORE_WINDOW = 60`,
> an 80k trial has 5 evals (100 rows) and the score uses the **last 3 trained evals**
> (40k/60k/80k), excluding the untrained `ctr=0` eval. A 40k trial has only 3 evals
> (60 rows), so the 60-row window would include the untrained eval — use ≥80k for BO,
> or lower `SCORE_WINDOW` for shorter trials.
It prints the best **feasible** config (efficiency, success, and each `--reward_*` value). If the study crashes or the machine reboots, re-run the **same** step-2 command — it resumes from where it stopped, not from trial 0.

Key flags: `--stage`, `--n-trials`, `--steps` (default **80000**), `--eval-episode-num`, `--eval-every` (forwarded to `dreamer.py`; default: omit → config default 20000; **lower it, e.g. 2000, for a fast smoke test**), `--seed`, `--margin` (allowed success drop, pts), `--timeout-per-trial` (sec, default **24 h** so a trial is never cut off unnoticed; a trial exceeding it is stopped and **scored on its partial eval data**, not discarded), `--run-baseline`, `--baseline-success <pct>` (skip the baseline run), `--logdir-root` (base for trial logdirs, default `./logdir` → trials land in `{logdir-root}/{study_name}/{study_name}_trial{NN}`), `--csv-dir` (default `./csv_logs/tune_stage{N}`), `--study-name` (default `reward_stage{N}`), `--dry-run`.

> **Per-trial budget reality:** a 40k-step trial takes ~140 min on this machine, so an **80k** trial (the recommended BO budget, see above) is ~280 min (~4.7 h); the timeout now defaults to **24 h** so a trial is never cut off unnoticed (no need to raise it for larger `--steps`; lower only for a hard cap). Trial logdirs are namespaced per study (`logdir/{study_name}/{study_name}_trial{NN}`), so re-runs never resume a stale checkpoint. The baseline is idempotent — it is reused if already scored, not re-trained.

> **CSV organization:** each trial emits ~9 CSVs, so trials write to a **per-stage subfolder** `csv_logs/tune_stage{N}/` (via `--csv-dir`, default `./csv_logs/tune_stage{stage}`) instead of flooding the main `csv_logs/`. Main/manual/validation runs keep writing to `csv_logs/`. The Streamlit dashboard has a **CSV-folder picker** in the sidebar to switch between `csv_logs/` (main runs, default) and any `tune_stage{N}/` subfolder.

> **Sanity test before a full study:** verify the full pipeline (Gazebo → subprocess → CSV scoring) with a short run before committing to 30 real trials. Pass a small `--eval-every` so each trial actually finishes fast (without it, the default 20000-step round makes even a 3k-step trial train a full round). Use `--study-name smoke` to isolate from the real study and `--logdir-root` so the output folder is obvious:
> ```bash
> python3 tune_reward.py --stage 1 --n-trials 2 --steps 4000 --eval-episode-num 2 \
>   --eval-every 2000 --study-name smoke --logdir-root ./logdir
> ```
> Trials land in `./logdir/smoke/smoke_trial000`, `…_trial001` (and `smoke_baseline_seed0` if a baseline runs); CSVs in `./csv_logs/tune_stage{N}/`. Delete the smoke artifacts before starting the real study — smoke logdirs, CSVs, and the db file are not reused by the real study (different `--study-name`), but cleaning up avoids confusion:
> ```bash
> rm -f tune_reward_stage1.db
> rm -rf logdir/smoke/ csv_logs/tune_stage1/
> ```

**Validation protocol (declares the winner):** take the printed weights and re-run at **full budget** with `dreamer.py` directly — 3 seeds on stage 1, then stages 2–4 with the same weights to test transfer:
```bash
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_tuned \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --reward_mode shaped \
  --reward_progress_scale <v> --reward_step_penalty <v> --reward_turn_penalty <v> \
  --reward_near_obstacle_scale <v> --reward_near_obstacle_sigma <v>
```
A single 40k proxy trial only *ranks* configs; the full-budget multi-seed run *confirms* the winner.

**Caveat — stage-1 over-fitting:** stage 1 is nearly obstacle-free, so weights tuned there can favor an aggressive progress reward that fails on cluttered stages. The progress search range is capped (0.1–3.0) and cross-stage validation is mandatory; if transfer is poor, re-tune on a cluttered stage (`--stage 3`).

## Path Plots

PNG overhead plots are generated per episode under `dreamerv3-torch/path_plots/{run_name}/`.

**Filename pattern:** `ep{episode:05d}_{outcome}.png` (e.g. `ep00001_timeout.png`, `ep00042_success.png`)
/
` DFT6YGHJJKL;P[]
**+-Generated whenever:** A* returns a valid path and the episode ends (success / collision / timeout). No toggle flag — always on. Disable during long training runs if I/O becomes a bottleneck.

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
+-
- 

+-'[]
*
live_chart.py` is no longer used.

Every `st.plotly_chart` / `st.dataframe` in the per-run sections (`_section_resource`, `_section_planner`) and the combined sections (`_section_bb`, `_section_wb`) passes a unique `key=` (e.g. `res_{run_name}_{y_col}`, `plan_{run_name}`, `bb_{col}`, `wb_{title}`). This avoids `StreamlitDuplicateElementId` when multiple runs are selected and rendered in a loop — **add a unique `key=` to any new chart/table** you introduce in those loops.

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
  - `--eval_every N` — the **evaluation interval in steps** (default `20000` = eval every 20k). Evaluation runs **once per `eval_every` steps** — `eval_every` is the single control (the hardcoded `% 4` multiplier was removed 2026-06-12; see [docs/evaluation_loop.md](docs/evaluation_loop.md)). **Lower** it (e.g. `--eval_every 5000`) for a finer learning curve; **raise** it for fewer eval pauses.
- **`eval_episode_num` guidance:**
  - `2` — smoke test / fast iteration only
  - `100` — preferred for real training and final result reporting
- **Eval overhead on long runs:** at the default `eval_every = 20000`, a 300k run is ~16 evals and a 600k run is ~31 evals. At `--eval_episode_num 100` that is ~1,600–3,100 eval episodes (and eval path-plots). Cut it further with a smaller `--eval_episode_num` and/or a larger `--eval_every`; conversely, lower `--eval_every` for a finer curve on short runs.

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

**GPU training — default reward (original sparse reward, baseline):**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/stage{n}_360_none_seed0_reward_default \
  --stage {n} --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --reward_mode default
```

**GPU training — shaped reward:**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/stage{n}_360_none_seed0_reward_shaped \
  --stage {n} --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --reward_mode shaped
```

**GPU training — shaped reward with tuned weights (post-BO validation):**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/stage{n}_360_none_seed0_reward_tuned \
  --stage {n} --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --reward_mode shaped \
  --reward_progress_scale <v> --reward_step_penalty <v> --reward_turn_penalty <v> \
  --reward_near_obstacle_scale <v> --reward_near_obstacle_sigma <v>
```

**IMU ablation — `full_imu` mode (branch `imu-observation`):**

Adds 2D linear acceleration from `/imu` on top of `full` odometry (7-dim `odometry` key). `/imu` is already published by the burger SDF on every stage — no Gazebo/SDF change needed. Use a **fresh logdir** (`.npz`/checkpoints are not compatible across odometry modes).

**CSVs/plots auto-organize into a per-stage subfolder.** `full_imu` runs write to `csv_logs/imu_stage{N}/` and `path_plots/imu_stage{N}/` automatically (mirrors how BO trials use `csv_logs/tune_stage{N}/`) — **no flags needed**. This is applied in `dreamer.py main()` and covers all nine CSVs (whitebox included) plus path-plots. It is skipped if you override `--csv_dir` / `--plots_dir`. The dashboard's sidebar folder picker lists `imu_stage{N}` automatically, just like `tune_stage{N}`.

Smoke test:
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/gpu_smoke_stage{n}_full_imu_seed0 \
  --stage {n} --lidar 360 --odometry_mode full_imu --seed 0 \
  --device cuda --steps 5000 --eval_episode_num 2
```

Full training:
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch

python3 dreamer.py \
  --configs turtle --task turtle \
  --logdir ./logdir/stage{n}_360_full_imu_seed0 \
  --stage {n} --lidar 360 --odometry_mode full_imu --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100
```

Verify `/imu` is live first (Gazebo running): `ros2 topic echo /imu --once` should show a populated `linear_acceleration`. The `full_imu` run logs `imu_enabled=True` and `sensor_config_id=lidar360_full_imu` in `resource_{run_name}.csv`, and `odometry_mode=full_imu` in `blackbox_{run_name}.csv` — all under the auto-created `csv_logs/imu_stage{N}/` subfolder.

Resource logging is **on by default** (`resource_logging: true` in `configs.yaml`) — no flag needed. To disable: add `--resource_logging False`.

Custom CSV output directory (e.g. to keep BO validation runs separate from tune trials):
```bash
  --csv_dir ./csv_logs/my_experiment
```

Replace `{n}` with the stage number (1–8) and `none` with `twist`, `delta`, `full`, or `full_imu` as needed.

## Git Rules

- **Never** use `git add .` or `git add -A`.
- **Never commit** generated output:
  - `logdir/` — episode archives and checkpoints
  - `csv_logs/` — training metrics CSVs
  - `path_plots/` — path visualization PNGs
  - `install/` — install artifacts
  - `log/` — ROS2/system log output
  - `*.pt`, `*.npz`, `*.jsonl`, `events.out.tfevents*`
  - `*.db`, `*.db-journal` — Optuna sqlite study files (`tune_reward_stage{N}.db`)
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

Nine CSV files per run, written under `dreamerv3-torch/csv_logs/` (or `config.csv_dir`). Train and eval episodes are separated — main CSVs contain only training episodes; eval episodes go to parallel eval CSVs:

| CSV file | Written by | Contents |
|---|---|---|
| `blackbox_{run_name}.csv` | Train env | Per-episode train metrics |
| `blackbox_eval_{run_name}.csv` | Eval env | Per-episode eval metrics (same schema) |
| `whitebox_{run_name}.csv` | `Logger.write()` | Step-based algorithm metrics (always train) |
| `planning_{run_name}.csv` | Train env | A* metrics for train episodes |
| `planning_eval_{run_name}.csv` | Eval env | A* metrics for eval episodes |
| `reward_{run_name}.csv` | Train env | Per-episode reward component sums |
| `reward_eval_{run_name}.csv` | Eval env | Reward component sums for eval episodes |
| `resource_{run_name}.csv` | Train env | Resource cost (on by default; `--resource_logging False` to disable) |
| `resource_eval_{run_name}.csv` | Eval env | Resource cost for eval (same flag) |

**`blackbox_*` columns:** `datetime, odometry_mode, stage, episode, outcome, steps_to_goal, path_directness, min_obstacle_dist, near_collisions, success_rate, collision_rate, rolling_success_rate_100, rolling_collision_rate_100, rolling_success_rate_500, rolling_collision_rate_500, episode_steps`. Episode numbering and rolling-window counters in each CSV are independent — cumulative rates and rolling rates in `blackbox_eval_*` reflect eval-only performance across all checkpoints.

> **`steps_to_goal` vs `episode_steps`:** `steps_to_goal` is the step count for **successful** episodes only (`-1` for collision/timeout — kept for backward compatibility). `episode_steps` (added 2026-06-12) is the total step count for **every** outcome: for `collision` rows it is the **time-to-collision**, for `timeout` it is ~`max_steps` (250), and for `success` it equals `steps_to_goal`. Combined with `actual_path_length` (planning CSV) it gives per-episode average speed (`m/step`) for any outcome. Read it by column name — CSVs written before the column was added simply lack the field.

**`whitebox_{run_name}.csv` columns:** `datetime, step, train_return, reward_variance, model_loss, actor_loss, value_loss, kl, prior_ent, post_ent, eval_return, eval_success_rate, eval_collision_rate, reward_variance_100, actor_entropy, model_grad_norm, actor_grad_norm, value_grad_norm, reward_loss, dyn_loss, rep_loss, ema_005, ema_095`

> **Whitebox is interval-based** (one row per `log_every` steps — turtle default `2e3`), written only when `model_loss` is present; expect far fewer rows than the per-episode blackbox. Use it for the **learning** diagnostics (losses, KL, entropies, returns).
>
> **Diagnostics added 2026-06-12** (the last 10 columns; all were already computed in `models.py`, so **zero extra training cost** — they were simply not being persisted). Appended at the **end** of the row, so older CSVs and existing readers are unaffected (the columns are just absent in pre-2026-06-12 files; read by name). Key ones:
> - **`actor_entropy`** — policy entropy. A collapse toward ~0 means the policy stopped exploring (a common cause of looping / getting stuck); the single most useful health signal. From [models.py](dreamerv3-torch/models.py) `actor_entropy`.
> - **`model_grad_norm` / `actor_grad_norm` / `value_grad_norm`** — gradient L2 norms; sudden spikes flag training instability.
> - **`reward_loss`** — isolated reward-head loss; if high, the world model can't predict reward and planning degrades. `dyn_loss` / `rep_loss` are the KL split (dynamics vs representation — posterior-collapse check).
> - **`ema_005` / `ema_095`** — DreamerV3 return-normalization percentiles (reward-scale drift; matters under shaped reward). Sourced from the upper-case `EMA_005`/`EMA_095` metric keys.
> - **`reward_variance_100`** — rolling last-100-episode reward variance. The original **`reward_variance` is cumulative (all-time)** and stays high forever because early-training chaos is never dropped; `reward_variance_100` is the honest "is it stable *now*?" signal. Both are kept.
>
> **`eval_success_rate` / `eval_collision_rate` (fixed 2026-06-09):** scored from the terminal **outcome label** (`tools.simulate` reads `info['outcome']` surfaced by `Turtle.step`), so they are correct under **shaped** reward and keep **timeout out of the collision count**. The old code compared episode reward to `100`/`-10`, which read `0` in shaped mode and folded timeouts into collisions. **For reported success/collision rates use `blackbox_eval_*` — it is the authoritative source in both reward modes.** CSVs logged before the fix (e.g. earlier tuner trials) still contain the buggy `0` values. See `docs/whitebox_data_validation.md`.

**`reward_*` columns:** `datetime, reward_mode, stage, episode, outcome, sum_terminal, sum_progress, sum_step_penalty, sum_turn_penalty, sum_near_obstacle, sum_total`. Written in **both** modes — in `default` mode the shaping columns are `0` and `sum_total == sum_terminal`.

**`planning_*` columns:** see **A* Planner-Based Path Efficiency** above; written on `planner-efficiency-metric` branch only.

**Routing mechanism:** File routing is done at `init_properties()` time based on `mode` — no runtime mode guards inside write methods. Train env opens `blackbox_{run_name}.csv`; eval env opens `blackbox_eval_{run_name}.csv`. The same resume logic (seed counters from existing CSV on restart) applies to both.

**Path plots:** Train episodes → `path_plots/{run_name}/`. Eval episodes → `path_plots/{run_name}_eval/`. Both are gitignored by `path_plots/`. Since eval fires every `eval_every` (default 20000) steps, a 300k-step run is ~16 eval checkpoints × `eval_episode_num` episodes — with `100` that is O(thousands) of eval PNGs. Reduce with a smaller `--eval_episode_num` / larger `--eval_every`, or disable A* in eval by patching `_write_planning_csv` if I/O becomes a bottleneck.

`run_name` is derived from the final component of `--logdir` in `make_env()`.

**CSV base directory:** all per-run CSVs are written under `config.csv_dir` (default `./csv_logs`, a config/CLI knob threaded through `make_env` → `Turtle` → `Env.init_properties` and `ResourceLogger`, **and into `tools.Logger` via `dreamer.py` for the whitebox CSV**). Only the base directory is configurable — the `{type}_{run_name}.csv` filenames, train/eval routing, and resume logic are unchanged. `tune_reward.py` sets `--csv_dir ./csv_logs/tune_stage{N}` so BO-trial CSVs stay in a per-stage subfolder; the dashboard's sidebar folder picker reads either `csv_logs/` or a subfolder.

> **Whitebox now honours `csv_dir` (fixed 2026-06-09).** Previously `tools.Logger` hardcoded the whitebox path to `./csv_logs/whitebox_{run}.csv`, so it split from the other CSVs on `--csv_dir`-redirected (tuner) runs. It now writes `{csv_dir}/whitebox_{run}.csv`, co-located with blackbox/planning/reward/resource. **Files written before the fix stay where they were** — Python loads `tools.py` once per process, so a run already in flight keeps the old path until it restarts; only new/next subprocesses pick up the change.

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

Enabled by default (`resource_logging: true` in `configs.yaml`). Pass `--resource_logging False` to turn it off (zero overhead when off).
Written to `dreamerv3-torch/csv_logs/resource_{run_name}.csv`, one row per episode.
Implementation: `dreamerv3-torch/envs/resource_logger.py`.

**Metric scope policy — primary columns are process-specific (DreamerV3 only):**

| Column | Scope | Notes |
|---|---|---|
| `cpu_percent_process` | Process only | `psutil.Process.cpu_percent()`; **can exceed 100%** — sum across all cores (PyTorch uses multiple threads; 150–400% is normal during GPU training) |
| `ram_used_mb_process` | Process only | `psutil.Process.memory_info().rss` |
| `gpu_memory_used_mb_process` | Process only | NVML `nvmlDeviceGetComputeRunningProcesses()` filtered by PID; **correct metric for thesis** — Gazebo uses OpenGL (not CUDA) so it does NOT appear here |
| `cpu_percent_system` | System-wide (context) | includes all desktop apps and Gazebo |
| `ram_percent_system` | System-wide (context) | includes all desktop apps and Gazebo |
| `gpu_util_percent_device` | Device-wide (context) | NVML has no per-process GPU util; includes desktop rendering load (Xorg, Chrome, VSCode) |
| `gpu_memory_used_mb_device` | Device-wide (context) | total device used — **typically ~600–1100 MB higher than `gpu_memory_used_mb_process`** due to Xorg (~370 MB), gnome-shell (~70 MB), VSCode (~88 MB), Chrome (~55 MB), gzserver/gzclient (~29 MB); do NOT use for thesis reporting |
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
- **`gpu_memory_used_mb_device` is significantly inflated on a desktop machine.** On this setup it runs ~600–1100 MB above `gpu_memory_used_mb_process` because NVML device memory includes all GPU processes (compute + OpenGL): Xorg (~370 MB), gnome-shell (~70 MB), VSCode (~88 MB), Chrome (~55 MB), gzserver/gzclient (~29 MB). **Use `gpu_memory_used_mb_process` for thesis reporting.** `gpu_memory_used_mb_device` is context only.
- **`cpu_percent_process` can exceed 100%** — `psutil` reports the sum across all CPU cores. PyTorch uses multiple threads (training, DataLoader, etc.); values of 150–400% are normal during active GPU training. To get per-core utilization, divide by `os.cpu_count()`.
- **Why CPU stays high despite CUDA training:** GPU handles only the model forward/backward pass. Everything else runs on CPU: ROS2 `spin_once` + LiDAR/odometry callbacks, reward computation, episode file I/O (`.npz` save/load), and PyTorch kernel dispatch overhead. For a small MLP-based model on LiDAR, the GPU finishes each batch in milliseconds — the actual bottleneck is Gazebo simulation speed (RTF ~4.5×) and ROS2 step latency, not the GPU. High `cpu_percent_process` with CUDA is expected and by design.
- **Gazebo resource cost is NOT captured** — `gzserver` and `gzclient` run as separate processes. Their CPU (typically 15–30% of system) and RAM are not in `cpu_percent_process` or `ram_used_mb_process`. For total experiment cost, the DreamerV3 process metrics must be combined with a separate measurement of the Gazebo processes.

**Dependencies:** `psutil` (already installed), `pynvml` / `nvidia-ml-py3` (installed). If either is missing, the affected fields are blank — training never crashes.

## Future Work

### Future perception modes
- `lidar` — current mode
- `depth` — depth camera (future)
- `lidar_depth` — combined (future)
- Depth camera and CNN integration are future work, separate from the current odometry ablation. Do not implement until explicitly scoped.