# Reward Shaping & Bayesian-Optimization (BO) Tuning

How the optional **shaped** reward works, how its five weights are tuned with **Optuna
Bayesian optimization** (`tune_reward.py`), how to **extract** the winning weights, and
how to **evaluate / validate** them. Grounded on
[turtle.py](../dreamerv3-torch/envs/turtle.py),
[tune_reward.py](../dreamerv3-torch/tune_reward.py), and
[export_tune_results.py](../dreamerv3-torch/export_tune_results.py). A Tagalog companion
(with a common-misconception table) is in
[reward_tuning_workflow_tagalog.md](reward_tuning_workflow_tagalog.md); the eval-loop /
`SCORE_WINDOW` details are in [evaluation_loop.md](evaluation_loop.md).

> **Key idea:** the five `--reward_*` weights are an **input/setting**, not a training
> output. Training produces a **model (`.pt`)** and **metrics**; the weights are chosen
> **only in the BO search**, then frozen for validation. A training run never "produces"
> weights.

---

## 1. Reward modes and the five shaping weights

`--reward_mode` selects the reward:

| Mode | Reward |
|---|---|
| `default` (default) | Original sparse reward, byte-for-byte: `+100` goal, `−10` collision, `−10` timeout, `0` otherwise. |
| `shaped` | Original terminal rewards **plus** the four additive per-step terms below. |

Shaped reward `= reward_default + progress + step_pen + turn_pen + near_obst`:

| RS | Term | Formula | **Default** (configs.yaml) | **BO search range** (`SEARCH_SPACE`) |
|---|---|---|---|---|
| RS-1 | Progress | `+scale · (prev_dist − curr_dist)` | `reward_progress_scale` = 1.0 | 0.1 – 3.0 (log) |
| RS-2 | Step penalty | `−c` per step | `reward_step_penalty` = 0.01 | 0.0 – 0.05 |
| RS-3 | Turn penalty | `−k · |ang_vel_cmd|` | `reward_turn_penalty` = 0.01 | 0.0 – 0.05 |
| RS-4 | Near-obstacle | `−scale · exp(−d_min/σ)` when `d_min < near_collision_threshold` (0.2 m) | `reward_near_obstacle_scale` = 0.1, `reward_near_obstacle_sigma` = 0.25 | scale 0.0 – 0.5, σ 0.1 – 0.5 |

> **The "BO search range" column is NOT a set of recommended/best values.** It is the
> **bounds Optuna explores**; the *best* value is what BO **finds inside** the range (one
> config per study — e.g. progress may land at ~0.7 within 0.1–3.0). The bounds are a
> **design choice**, not a proven optimum, with a rationale: the progress range is
> **capped at 0.1–3.0** to avoid stage-1 over-fitting (an aggressive progress reward that
> fails on cluttered stages — see CLAUDE.md), and the others are kept **mild** so shaping
> stays gentle. To change them, edit `SEARCH_SPACE` ([tune_reward.py:53](../dreamerv3-torch/tune_reward.py#L53))
> and re-tune. RS-4's activation gate is the **shared** `near_collision_threshold`
> (tightened to **0.2 m** on 2026-06-13 — see CLAUDE.md *Collision detection thresholds*).

One **set** of these five numbers is a "config"/"weights".

---

## 2. The BO search — `tune_reward.py` is an orchestrator, not a trainer

`tune_reward.py` **wraps** `dreamer.py`; it has no training loop of its own. Each Optuna
trial:

1. The **TPE sampler** suggests 5 weights (`suggest_weights`, within `SEARCH_SPACE`).
2. It builds and runs a `dreamer.py … --reward_mode shaped --reward_* …` **subprocess**
   on a short proxy budget.
3. It **scores** the run from the eval CSVs (`score_run`).
4. It returns efficiency + a success constraint to Optuna, which updates its surrogate
   model and proposes the next trial.

**Objective (constrained efficiency):** maximize eval `planner_path_efficiency` subject
to eval `success_rate ≥ baseline_success − margin` (default margin 5 pts). The constraint
is enforced via `TPESampler(constraints_func=…)`.

**Scoring window:** `score_run` reads the **last `SCORE_WINDOW = 60` eval rows**
(`planning_eval_*` for efficiency, `blackbox_eval_*` for success). With
`--eval-episode-num 20` and `eval_every = 20000`, 60 rows = the **last 3 evals** —
the most-converged checkpoints, excluding the untrained `ctr=0` eval.
- **Use `--steps ≥ 80000`** (this project uses **100000**). A 100k run scores
  60k/80k/100k; an 80k run scores 40k/60k/80k. A **40k** run has only 3 evals incl.
  `ctr=0`, so the 60-row window would reach the **untrained** eval and pollute the score.
  See [evaluation_loop.md](evaluation_loop.md).

**Persistence:** the study is saved to `tune_reward_stage{N}_{mode}.db` (sqlite,
gitignored) with `load_if_exists=True` — re-running the **same** command **resumes** the
study (it does not restart from trial 0). `--n-trials 30` **adds** 30 trials per
invocation (it is not a total cap).

---

## 3. Tuning under an odometry mode (`--odometry-mode`)

The observation space materially changes what the policy can learn, so weights tuned
under `none` may not be optimal under `full_imu` — **tune under the mode you will
deploy.** The mode **always namespaces** the study/db/csv/plots (suffix `_{mode}`,
**including `_none`**):

| `--odometry-mode` | study / db | trial CSVs |
|---|---|---|
| `none` (default) | `reward_stage{N}_none` / `tune_reward_stage{N}_none.db` | `csv_logs/tune_stage{N}_none/` |
| `full_imu` | `reward_stage{N}_full_imu` / `tune_reward_stage{N}_full_imu.db` | `csv_logs/tune_stage{N}_full_imu/` |

So studies for the same stage but different modes never collide.

---

## 4. Running the search (two terminals)

**Terminal 1 — Gazebo once** (stays up for the whole study; headless for eGPU stability):
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/turtle_stage4.py gui:=false
```

**Terminal 2 — the tuner** (this project's config: stage 4, `full_imu`, **30 trials,
100000 steps, 20 eval episodes**):
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 tune_reward.py --stage 4 --odometry-mode full_imu \
  --n-trials 30 --steps 100000 --eval-episode-num 20
```
It auto-runs a **default-reward baseline first** (to set the success-rate floor) unless
`--baseline-success <pct>` is given. Trials run **sequentially** (one Gazebo, one GPU).
If it crashes/reboots, re-run the **same** command to resume.

---

## 5. Extracting the winning weights (after the trials)

Optuna picks the best **feasible** trial (highest efficiency whose success did not drop
more than the margin). **Do not** re-run `tune_reward.py --n-trials …` just to peek — that
**adds** more trials. Use one of:

**Option A — `export_tune_results.py`:**
```bash
python3 export_tune_results.py --stage 4 --odometry-mode full_imu
```
Writes (under `csv_logs/tune_stage4_full_imu/`):
- `tune_trials_stage4.csv` — **all** trials + each trial's 5 weights.
- `tune_best_stage4.csv` — the **single best feasible** config (one row).

**Option B — read the study DB directly:**
```bash
python3 - <<'PY'
import optuna
s = optuna.load_study(study_name="reward_stage4_full_imu",
                      storage="sqlite:///tune_reward_stage4_full_imu.db")
done = [t for t in s.trials if t.value is not None]
feas = [t for t in done if t.user_attrs.get("constraint", (1.0,))[0] <= 0]
best = max(feas or done, key=lambda t: t.value)
print("best trial", best.number, "eff", round(best.value, 4),
      "succ", best.user_attrs.get("success"))
for k, v in best.params.items():
    print(f"  --{k} {v:.6g}")
PY
```

---

## 6. Validating the winner (this declares the result)

A short proxy trial only **ranks** configs; the full-budget multi-seed run **confirms**
the winner. **Insert the single best 5-weight set** below — same weights everywhere, and
run under the **same odometry mode** you tuned (here `full_imu`).

**Phase 2 — 3 seeds, full budget, same stage + mode:**
```bash
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage4_360_full_imu_seed0_reward_tuned \
  --stage 4 --lidar 360 --odometry_mode full_imu --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --reward_mode shaped \
  --reward_progress_scale <v> --reward_step_penalty <v> --reward_turn_penalty <v> \
  --reward_near_obstacle_scale <v> --reward_near_obstacle_sigma <v>
```
Repeat for seeds `1` and `2` (**same** 5 weights). For the report, take the
**mean ± std** of `success_rate` and `planner_path_efficiency` from `blackbox_eval_*` /
`planning_eval_*` — **average the metrics, not the weights**. Use `blackbox_eval_*` for
success/collision (see [whitebox_data_validation.md](whitebox_data_validation.md)).

**Phase 3 — transfer:** run the **same** 5 weights on other stages (a **fresh logdir**
per stage → from-scratch training; only the reward *recipe* transfers, **not** the `.pt`).
This is **not** curriculum learning — no model is carried across stages.

---

## 7. Cheat sheet

| Thing | How many | Averaged? | Source |
|---|---|---|---|
| Reward weights (5 numbers) | **1 set** | ❌ | BO search (Phase 1) |
| 3-seed metrics (success, efficiency) | 3 | ✅ mean ± std | Validation (Phase 2) |
| Weights carried to other stages | **1 set** (same) | ❌ | still from BO |
| `.pt` checkpoint | 1 per run | — | training output; **not** carried forward |

**Flow:** 30 BO trials → **1 best feasible config** → 3-seed validation on the tuned stage
(average the **metrics**) → if good, the **same 5 weights** on other stages for the
transfer test.
