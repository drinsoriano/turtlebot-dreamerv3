# Reward Shaping: default, additive `shaped`, and potential-based `pbrs`

This project has **three** reward modes, selected with `--reward_mode`. All three keep the
original sparse **terminal** reward exactly; they differ only in what (if anything) they add on
top of it. The mode is grounded in
[envs/turtle.py](../dreamerv3-torch/envs/turtle.py) (`get_reward_and_done`) and
[configs.yaml](../dreamerv3-torch/configs.yaml). For the Bayesian-optimization tuning of the
`shaped` weights see [reward_tuning_workflow.md](reward_tuning_workflow.md).

> **Key rule for every mode:** the terminal reward is **never** changed, and **A\* is never fed
> into the reward**. A\* (`planner_path_efficiency`) stays a *post-hoc evaluation metric* only —
> using it inside the reward would leak the optimal path and defeat the purpose of *learning* to
> navigate. See [Why A\* is not used as a reward](#why-a-is-not-used-as-a-reward).

| `--reward_mode` | Per-step reward |
|---|---|
| `default` (default) | sparse only: `+100` goal, `−10` collision, `−10` timeout, `0` otherwise |
| `shaped` | `default` **+** four additive hand-weighted terms (RS-1…RS-4) |
| `pbrs` | `default` **+** one potential-based shaping term (policy-invariant) |

---

## 1. `default` — the original sparse reward

```
reward = +100   if the robot reaches the goal region (distance < REACH_TRESHOLD = 0.4 m)
       = −10    if it collides       (min LiDAR < COLISION_TRESHOLD = 0.13 m)
       = −10    if it times out      (step_counter ≥ max_steps − 1)
       = 0      otherwise
```

This is the unmodified baseline. It is *sparse*: the agent only learns from the rare terminal
events, so credit assignment over a long episode is slow. `default` is byte-for-byte unchanged
and is the control condition for every comparison.

## 2. `shaped` — additive, hand-weighted shaping

`shaped` adds four dense per-step terms to the **unchanged** terminal reward:

```
reward = reward_default + progress + step_pen + turn_pen + near_obst
```

| RS | Term | Formula | Knob (default) |
|---|---|---|---|
| RS-1 | Progress | `+scale · (prev_dist − curr_dist)` | `reward_progress_scale` (1.0) |
| RS-2 | Step penalty | `−c` each step | `reward_step_penalty` (0.01) |
| RS-3 | Turn penalty | `−k · \|ang_vel_cmd\|` | `reward_turn_penalty` (0.01) |
| RS-4 | Near-obstacle | `−scale · exp(−d_min/σ)` when `d_min < near_collision_threshold` (0.2 m) | `reward_near_obstacle_scale` (0.1), `reward_near_obstacle_sigma` (0.25) |

This densifies the reward and works well in practice, but it is **not policy-invariant**: each
weight tilts *which* behaviour is optimal (e.g. too strong a progress term discourages the
detours that cluttered stages require). That is exactly why the weights must be **tuned** (BO)
and why the search caps progress at 0.1–3.0 and keeps the rest mild — see
[reward_tuning_workflow.md](reward_tuning_workflow.md). The weights are a *design knob*, not a
theorem.

## 3. `pbrs` — relative distance + angular potential-based shaping

PBRS adds a **single** term derived from a *potential function* `Φ(s)`:

```
r_pbrs = pbrs_scale · ( pbrs_gamma · Φ(s_{t+1}) − Φ(s_t) )
reward = reward_default + r_pbrs

Φ(s)   = −( pbrs_distance_weight · d_norm  +  pbrs_angle_weight · a_norm )
d_norm = min(distance_to_goal / pbrs_distance_scale, 1.0)     ∈ [0, 1]   (0 at the goal)
a_norm = (1 − cos(angle_to_goal)) / 2                         ∈ [0, 1]   (0 facing the goal)
```

- `distance_to_goal` and `angle_to_goal` are the **same relative goal distance/bearing already
  computed in `Env.get_state()`** (from `/odom` pose + the spawned goal) and already fed to the
  policy — PBRS reuses them, it does **not** read the raw pose and does **not** read A\*.
- `Φ ≤ 0`, rising toward `0` as the robot gets **closer** and **better aligned** to the goal. So
  `r_pbrs > 0` for goal-directed progress and `< 0` for moving away — a dense, informative signal.
- It is computed on the **transition `s_t → s_{t+1}`**: `prev_distance_to_target` /
  `prev_angle_to_target` are `s_t`; the fresh `get_state()` values are `s_{t+1}`. (This mirrors how
  the `shaped` RS-1 progress term already uses `prev_distance_to_target`.)

### Knobs (in `configs.yaml`, all CLI-overridable)

| Knob | Default | Meaning |
|---|---|---|
| `pbrs_scale` | 1.0 | overall weight on the shaping term |
| `pbrs_distance_weight` | 1.0 | weight of the distance component of `Φ` |
| `pbrs_angle_weight` | 0.2 | weight of the heading-alignment component of `Φ` |
| `pbrs_distance_scale` | 5.0 | distance normaliser (m); **fixed**, not stage-aware (see below) |
| `pbrs_gamma` | 0.997 | PBRS discount; **keep equal to the agent `discount` (0.997)** |

**Why `pbrs_distance_scale = 5.0` is fixed (not stage-aware).** The robot resets at the origin and
goals are sampled within each arena, so the max start→goal distance is ≈ 2.83 m (stages 1–4),
≈ 4.24 m (stages 5/8), and up to ≈ 7 m (stage 6). A single fixed scale of 5.0 m keeps `Φ`
**comparable across stages** and keeps `d_norm` in its informative linear band on stages 1–4 (it
never saturates at 1.0 there). A per-stage scale would fill `[0,1]` on every stage but make `Φ`
incomparable across stages and add a stage→scale lookup — not worth it for the initial study.
Lower the scale for a steeper near-goal gradient; it stays a single CLI knob.

**Guards.** If any of the current/previous distance or angle is NaN/inf, that step's `r_pbrs` is
set to `0.0` (the shaping is skipped, training never crashes). The literal `F = γ·Φ(s') − Φ(s)`
is applied on every step **including terminal** steps; the strict-invariance convention
`Φ(terminal)=0` is *not* enforced because the ±terminal reward dominates terminal steps anyway.

### Why PBRS is more theoretically grounded than `shaped`

PBRS comes with a **policy-invariance theorem** (Ng, Harada & Russell, *"Policy invariance under
reward transformations"*, ICML 1999): if the extra reward has the form
`F(s, s') = γ·Φ(s') − Φ(s)` for **any** potential `Φ` and the **same discount `γ`** the agent
uses, then the set of optimal policies is **provably unchanged**. The shaping can only speed up
*learning* (denser gradient toward the goal); it cannot change *what* the optimal behaviour is.

The additive `shaped` terms have **no such guarantee** — each weight changes the optimal policy,
which is why they need careful tuning and a success-rate guard. PBRS's only requirements are
(1) the reward is a difference of potentials and (2) `pbrs_gamma == discount` (here both `0.997`).
That is the whole reason this mode exists: a dense reward you can add **without** worrying that it
silently changes the task — a cleaner, principled baseline for "does denser reward help path
efficiency?" than the hand-weighted `shaped`.

### Why A\* is not used as a reward

A\* gives the optimal collision-free path for each episode's start/goal pair, so it is tempting as
a reward signal. It is deliberately **excluded** from every reward mode because:

- **It leaks the answer.** Rewarding closeness to the A\* path tells the agent the solution
  instead of making it *learn* navigation from LiDAR — the thesis question is whether DreamerV3
  *learns* efficient navigation, not whether it can imitate a planner.
- **It needs the global map.** A\* uses `STAGE_ARENAS` geometry; the policy only ever sees LiDAR +
  relative goal. Feeding A\* into the reward would smuggle privileged global information into a
  supposedly local-sensing agent.
- **It would invalidate the metric.** `planner_path_efficiency` (= planned/actual length) is the
  *primary evaluation metric*. If A\* were also in the reward, the metric would measure how hard we
  optimised toward A\*, not genuine learned efficiency.

So A\* stays strictly **post-hoc / diagnostic** (`planning_{run}.csv`, path-plot overlays). PBRS's
potential uses only the relative goal distance/angle, which the agent already observes.

---

## 4. Component logging

Per-episode component **sums** go to `csv_logs/reward_{run}.csv` (eval → `reward_eval_{run}.csv`),
written in **all three** modes:

```
datetime, reward_mode, stage, episode, outcome,
sum_terminal, sum_progress, sum_step_penalty, sum_turn_penalty, sum_near_obstacle, sum_total,
sum_pbrs, sum_pbrs_phi_current, sum_pbrs_phi_next,
sum_pbrs_distance_component, sum_pbrs_angle_component
```

The five `sum_pbrs*` columns are **appended at the end** so existing readers and pre-PBRS CSVs are
unaffected (read by column name; older files just lack them). Columns that a mode does not use are
`0` (in `default` all shaping columns are `0`; in `shaped` the five `sum_pbrs*` are `0`; in `pbrs`
the four additive columns are `0`). `sum_total` always equals the actually-applied per-episode
reward. The dashboard does not read this CSV, so the schema change is dashboard-safe.

## 5. Recommended experiment matrix

Compare the three modes on a **fresh logdir each**, holding everything else constant (stage, seed,
budget, odometry mode, RTF, collision threshold). Logdir convention:
`stage{N}_360_{odom}_seed{S}_reward_{default|shaped|pbrs}`.

| Run | `--reward_mode` | Purpose |
|---|---|---|
| baseline | `default` | sparse-reward control |
| additive | `shaped` (tuned weights) | hand-weighted dense reward |
| PBRS | `pbrs` | policy-invariant dense reward |

- **Primary metric:** eval `planner_path_efficiency` (`planning_eval_{run}.csv`).
- **Secondary:** `path_directness` (`blackbox_eval_{run}.csv`).
- **Guards (must not regress):** `success_rate`, `collision_rate` from `blackbox_eval_{run}.csv`
  (the authoritative source in every reward mode — see
  [whitebox_data_validation.md](whitebox_data_validation.md)).
- **Robustness:** repeat each mode over seeds `0,1,2` and report **mean ± std** of the metrics;
  optionally test transfer to a cluttered stage (e.g. 3 or 4).

Example (stage 1, all three; swap `{n}`/seed as needed):

```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch
# default
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_default \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 --reward_mode default
# shaped (insert tuned weights)
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_shaped \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 --reward_mode shaped
# pbrs
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_pbrs \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 --reward_mode pbrs
```

Smoke test any mode by dropping to `--steps 5000 --eval_episode_num 2` and a `gpu_smoke_*` logdir.
