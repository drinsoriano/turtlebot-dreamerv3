# Step ↔ Time ↔ Distance — What a "Step" Actually Is

This note pins down the relationship between a training **step**, simulated
**time**, the **distance** the robot travels, and a **wheel rotation** — because
it is easy to conflate them and get the numbers wrong by an order of magnitude.

> **Framing:** training is counted in **steps** (the reinforcement-learning
> convention). Meters are a **derived** quantity and are **not** a fixed multiple
> of steps. Report training budget in steps; report navigation quality in meters.

---

## 0. The one misconception to kill first — a step is TIME, not DISTANCE

> A **step** is one *decision tick* — a fixed slice of *time* during which the
> robot moves at **whatever speed the agent just commanded**. It is **not** a
> fixed distance.

The trap is to picture a step as "the robot moved one fixed unit forward."
It is actually "one moment where the agent made a decision." Distance is a
*consequence* of that decision, not the definition of the step.

Speed varies **by design**, because **speed is the action**
([turtle.py:850](../dreamerv3-torch/envs/turtle.py#L850)):

```python
linear_vel  = abs(action[0]) * 0.1     # the agent CHOOSES this every step
angular_vel = action[1] * 2 * 0.1
```

| Agent commands | linear_vel | distance that step (~1 s) |
|---|---|---|
| `action[0] = 1.0` (full throttle) | 0.10 m/s | ~0.10 m |
| `action[0] = 0.5` (half) | 0.05 m/s | ~0.05 m |
| `action[0] = 0.0` (stop / just turn) | 0.00 m/s | ~0 m |

Same step, same duration — **different distance, because the agent pressed the
throttle differently.** If speed *couldn't* vary, the agent could never slow
down near an obstacle, never stop at the goal, never turn in place. The varying
speed is not imprecision — **it is the entire thing DreamerV3 is learning.**

**Analogies:**

- **Metronome.** A step is one *tick*. The beat is steady (time). How far you
  walk between two ticks depends on whether you sprint or stroll — your choice,
  not the metronome's.
- **Video-game frame (60 FPS).** Each frame is a fixed slice of time. How far
  your character moves in one frame depends on how hard you push the stick. The
  frame defines *when you decide*, not *how far you go*.

---

## 1. TL;DR

```
1 step     = one decision tick ≈ 0.6–1.4 s sim time,  ~0.065 m travelled (≤ ~0.14 m)
1 episode  = up to 250 steps,                          up to ~26 m of travel (timeout)
success reaches the goal in ~30 steps ≈ 1.9 m on average
```

The step→meter factor is a **measured distribution, not a constant.**

---

## 2. Why reinforcement learning counts in steps, not meters

An RL agent lives on a discrete decision cycle — the MDP timestep. Every step it
does exactly one thing: observe → act → receive reward. So the **step is the
natural clock of the algorithm**, and everything about *learning* is indexed by it:

- **Sample efficiency** — the field reports learning curves as *reward/success
  vs steps* (e.g. Atari "200M frames", MuJoCo "1M steps"). This project's
  `--steps` budget and the whitebox `step` column follow that convention.
- **Reproducibility** — steps are invariant to how fast the simulator runs.
  Meters depend on the velocity scaling (`*0.1`), the per-step `dt`, and Gazebo
  real-time factor (RTF) — all of which can change. Two machines at different RTF
  do the *same* learning in the *same* steps but different wall-clock.
- **Hardware-agnostic** — a sim step and a real-robot control step are the same
  abstraction, even though the meters-per-step differ.

---

## 3. The four "clocks" — only one conversion is a true constant

| Conversion | Fixed? | Why |
|---|---|---|
| wheel rotation ↔ **meters** | ✅ fixed | wheel circumference (0.207 m) — pure geometry |
| **step** ↔ meters | ❌ varies | distance = commanded speed × sim-`dt`; speed is chosen each step |
| step ↔ wheel rotation | ❌ varies | inherits the step↔meter variability |
| step ↔ sim-time | ≈ roughly | ~0.6–1.4 s, depends on `spin_once`/RTF |

**Wheel rotation is not in the control loop.** The policy commands `/cmd_vel`
(linear + angular **velocity**). Gazebo's differential-drive plugin then converts
that velocity into wheel angular speeds internally (`v = ω_wheel × r`, with
`r = 0.033 m`). At the 0.1 m/s cap, ω ≈ 3.0 rad/s ≈ 0.48 rev/s ≈ one rotation
every ~2.1 s. The agent never sees or commands wheel rotations — they are a
*downstream physical consequence*, one layer below `cmd_vel`.

---

## 4. Code-grounded constants

| Fact | Source |
|---|---|
| Max linear speed = 0.10 m/s (`abs(action[0]) * 0.1`) | [turtle.py:850](../dreamerv3-torch/envs/turtle.py#L850) |
| Max angular speed = 0.20 rad/s (`action[1] * 2 * 0.1`) | [turtle.py:851](../dreamerv3-torch/envs/turtle.py#L851) |
| Max steps/episode = 250 (`time_limit`, `action_repeat: 1`) | [configs.yaml:38](../dreamerv3-torch/configs.yaml#L38), [110](../dreamerv3-torch/configs.yaml#L110) |
| `path_length` = cumulative Euclidean trajectory length | [turtle.py:300](../dreamerv3-torch/envs/turtle.py#L300) |
| One step ≈ 0.6–1.4 s **sim time** (derived: 0.14 m ÷ 0.1 m/s) | inferred from data |
| Wheel diameter = 0.066 m, separation = 0.160 m | [model.sdf:379-380](../turtlebot3_gazebo/models/turtlebot3_burger/model.sdf#L379) |
| 1 wheel rotation = π × 0.066 ≈ **0.207 m** (fixed; no-slip) | geometry |

---

## 5. Empirical step→meter conversion (measured, not assumed)

Measured from stage-1 BO-trial CSVs (~38k episodes), joining the planning CSV
`actual_path_length` with the blackbox CSV `steps_to_goal`:

| Outcome | n | mean path (m) | max path (m) |
|---|---|---|---|
| success | 36,707 | 1.94 | 17.84 |
| collision | 660 | 3.95 | 19.23 |
| **timeout** | 385 | **14.27** | **26.07** |

- **meters/step** (success episodes): mean ≈ **0.065**, median ≈ 0.066, max ≈ 0.14
- **steps_to_goal** (success): mean ≈ 30, max 247

This is a *distribution*, not a single number — the spread (0.019–0.14 m/step)
is exactly the agent choosing different speeds.

Reproduce:

```python
import pandas as pd, glob
for plf in glob.glob("csv_logs/tune_stage1/planning_reward_stage1_trial*.csv"):
    bbf = plf.replace("planning_", "blackbox_")
    pl, bb = pd.read_csv(plf), pd.read_csv(bbf)
    m = pl.merge(bb[["episode", "outcome", "steps_to_goal"]], on="episode")
    s = m[(m.outcome == "success") & (m.steps_to_goal > 0)]
    apl = pd.to_numeric(s.actual_path_length, errors="coerce")
    print((apl / s.steps_to_goal).mean())   # ≈ 0.065
```

---

## 6. Why the naive "2.5 m per episode" figure is wrong

A tempting back-of-envelope is: *step = one 10 Hz LiDAR frame = 0.1 s; at max
0.1 m/s that is 0.01 m/step; 250 steps = 2.5 m.* Every number after the first
assumption is fine — but a step is **not** one LiDAR frame. Two
`rclpy.spin_once(timeout_sec=0.5)` calls per step plus Gazebo RTF make each step
**~0.6–1.4 s** of simulated motion, so the robot covers **~0.065 m/step**, and a
250-step timeout episode covers up to **~26 m** — about 10× the naive estimate,
as the data above confirms.

---

## 7. Use the right unit per question — and why `planner_path_efficiency` is robust

| Question | Right unit |
|---|---|
| How much compute did learning take? | **steps** |
| Did the robot take an efficient route? | **meters** (`actual_path_length`, A\* path length) |
| How fast did it reach the goal? | steps-to-goal (≈ seconds, since `dt` is ~constant) |

The headline navigation metric sidesteps the whole conversion:

```
planner_path_efficiency = planned_path_length / actual_path_length     # meters / meters
```

Because it is a **dimensionless ratio (m/m)**, the step↔meter factor cancels — it
does not matter whether a step is 0.065 m or 0.01 m. That is exactly why it is
the trustworthy thesis metric. In practice efficiency spans roughly **0.05** (a
long timeout that wanders ~14–26 m against a ~1.5 m optimal) to **~0.95** (a
near-optimal success).

See [methodology.md](methodology.md) for how this metric is defined and reported,
and the **A\* Planner-Based Path Efficiency** section of `CLAUDE.md` for the CSV
schema and `planner_status` values.

> A Tagalog walkthrough of the core misconception (with the metronome /
> speedometer / video-game-frame analogies) is in
> [step_distance_relationship_tagalog.md](step_distance_relationship_tagalog.md).
