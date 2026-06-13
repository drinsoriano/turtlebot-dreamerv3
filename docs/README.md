# DreamerV3-Based Autonomous Mobile Robot Navigation — Research Documentation

This folder documents the **current DreamerV3-based autonomous mobile robot
navigation** implementation in this repository. A TurtleBot3 (burger) robot
learns to reach randomly spawned goals while avoiding obstacles inside a
ROS2 / Gazebo simulation, using the model-based reinforcement learning agent
**DreamerV3** (PyTorch port).

All documentation here is derived directly from the source code under
`dreamerv3-torch/`, with the evidence-based code audit in
[implementation_audit.md](implementation_audit.md) as the **primary source of
truth**. Where a statement is an interpretation rather than a literal code fact,
it is marked **inferred from implementation**. Where a component is not present,
it is marked **not found in the current implementation**.

## Scope and exclusions

- **In scope:** the DreamerV3 navigation agent, its ROS2/Gazebo environment
  wrapper, observation/action construction, reward and termination logic,
  world-model/actor-critic learning, replay, logging, evaluation, and the
  monitoring dashboard.
- **Out of scope (excluded):** the legacy model-free deep RL implementations
  found under `model_free/` (TD3, DDPG, SAC) and the associated root-level
  scripts. These are treated as **legacy / out-of-scope** and are **not** used
  as components or comparison algorithms anywhere in this documentation.
- The Gazebo world directories named `turtlebot3_dqn_stageN` are **map/world
  names only** — they are simulation arenas, not an algorithm. No DQN code is
  used by the DreamerV3 pipeline.
- **Supported arena stages: 1 to 8.** The goal sampler and the A* arena geometry
  both cover stages 1 to 8 (the stage 7 and 8 goal sampler was added 2026-06-09).
  Each stage requires its matching Gazebo launch file and a `--stage` argument
  that matches the running simulation.

## Documentation index

| Document | Purpose |
|---|---|
| [implementation_audit.md](implementation_audit.md) | **Evidence base.** Static, read-only audit of what is actually implemented, with per-finding confidence labels. The source of truth for every other document. |
| [methodology.md](methodology.md) | Thesis-ready Chapter 3 methodology grounded strictly on the audit: research design, environment, observation and action spaces, the DreamerV3 framework, training and evaluation procedures, variables, metrics, analysis, limitations, reproducibility, and a methodology flowchart. |
| [research_process_flow.md](research_process_flow.md) | End-to-end research process: problem definition through metric analysis, with training and evaluation flows. |
| [system_architecture.md](system_architecture.md) | Software/system architecture: simulation, environment wrapper, agent internals, replay, logging, dashboard. |
| [conceptual_framework.md](conceptual_framework.md) | Thesis conceptual framework: input variables, learning mechanism, navigation behavior, performance outcomes. |
| [observation_and_action_space.md](observation_and_action_space.md) | Exact observation keys/shapes and the continuous action space, with the action-to-motion mapping. |
| [training_and_evaluation_pipeline.md](training_and_evaluation_pipeline.md) | Episode loop, model updates, evaluation protocol, and CSV metric logging (black-box and white-box). |
| [whitebox_data_validation.md](whitebox_data_validation.md) | Read-only audit of black-box vs white-box CSV logging: correctness, consistency, issues found, and how to use white-box data in results. |
| [reward_tuning_workflow.md](reward_tuning_workflow.md) | The shaped reward + Bayesian-optimization (Optuna) tuning workflow: the five weights and their **BO search ranges** (bounds, not best values), how `tune_reward.py` wraps `dreamer.py` with a constrained efficiency objective and the `SCORE_WINDOW` scoring, `--odometry-mode` study namespacing, how to **extract** the winning weights (`export_tune_results.py` / study DB), and the 3-seed + transfer **validation** protocol. |
| [reward_tuning_workflow_tagalog.md](reward_tuning_workflow_tagalog.md) | **(Tagalog)** Companion of reward_tuning_workflow.md: BO search vs validation, bakit 1 set lang ng weights, 3-seed validation (average ng metrics), transfer sa stage 2–4, at bakit hindi ito curriculum learning. |
| [step_distance_relationship.md](step_distance_relationship.md) | What a training **step** actually is: step ↔ time ↔ distance ↔ wheel-rotation. Why a step is a tick of *time* (not distance), why meters-per-step varies (speed is the action), code-grounded constants, and the measured ~0.065 m/step / ~26 m-per-episode figures. Steps-first framing; explains why `planner_path_efficiency` (dimensionless) is robust. |
| [step_distance_relationship_tagalog.md](step_distance_relationship_tagalog.md) | **(Tagalog)** Companion ng step_distance_relationship.md: "ang step ay ORAS, hindi layo", bakit nag-iiba ang bilis (ang bilis ang aksyon), at mga analohiya (metronome, gas pedal, video-game frame). |
| [training_convergence.md](training_convergence.md) | How long to train: why success rate and path efficiency plateau early (stage-1 evidence), diminishing returns vs. cost, why efficiency improves via reward shaping (not more steps), how to choose `--steps` per stage, and the 🎯 Convergence dashboard tab. Includes a Tagalog buod. |
| [evaluation_loop.md](evaluation_loop.md) | The training/evaluation loop and every parameter that drives it (`prefill`, `eval_every`, `eval_episode_num`, `steps`, `train_ratio`, …): prefill → train/eval cycle, why eval actually runs every 20k steps (the `% 4` artifact), why training now stops exactly at `--steps` (the 2026-06-13 break fix — no wasted post-final-eval tail), and the recommendation to use plain `eval_every` for a finer curve. Mermaid diagrams + Tagalog buod. |
| [research_framework_summary.md](research_framework_summary.md) | Condensed, thesis-ready synthesis suitable for a Chapter 1 conceptual framework or Chapter 3 methodology. |

## Repository inspection summary

The documentation is based on the following files actually used by the
DreamerV3 navigation pipeline (all paths relative to the repository root):

| File | Role in the DreamerV3 pipeline |
|---|---|
| `dreamerv3-torch/dreamer.py` | Training/evaluation entry point. Defines the `Dreamer` agent, `make_env`, the config/CLI merge, the prefill, the train/eval loop, and checkpointing. |
| `dreamerv3-torch/configs.yaml` | Default + `turtle` configuration blocks: world-model (RSSM), actor, critic, reward/continue heads, training, behavior, and reward-shaping knobs. |
| `dreamerv3-torch/envs/turtle.py` | The ROS2 environment: `Env(Node)` (publishes `/cmd_vel`, subscribes `/scan` and `/odom`, plus `/imu` in `full_imu` mode) and the `Turtle(gym.Env)` wrapper that defines the observation and action spaces, reward, termination, goal sampling, and CSV logging. |
| `dreamerv3-torch/models.py` | `WorldModel` (encoder, RSSM dynamics, decoder/reward/continue heads) and `ImagBehavior` (actor and critic over imagined rollouts). |
| `dreamerv3-torch/networks.py` | Neural building blocks: `RSSM`, `MultiEncoder`/`MultiDecoder`, `MLP`, `GRUCell`. |
| `dreamerv3-torch/tools.py` | `simulate()` environment driver, disk-backed replay (`load_episodes`, `sample_episodes`, `from_generator`, `save_episodes`), and the `Logger` that writes the white-box CSV. |
| `dreamerv3-torch/envs/stage_map.py` | Stage arena geometry and the A* planner (`get_grid`, `astar_plan`, `STAGE_ARENAS`) used for the post-hoc path-efficiency metric. |
| `dreamerv3-torch/envs/path_viz.py` | Per-episode overhead path-plot rendering. |
| `dreamerv3-torch/envs/resource_logger.py` | Per-episode compute-resource logging (CPU/RAM/GPU via psutil/NVML). |
| `dreamerv3-torch/envs/wrappers.py` | Environment wrappers (e.g. `UUID`). |
| `dreamerv3-torch/dashboard/app.py` | Streamlit monitoring dashboard that reads the per-run CSV logs. |
| `dreamerv3-torch/tune_reward.py` | Standalone Optuna Bayesian-optimization of the shaped-reward weights (orchestrates `dreamer.py` as a subprocess; does not alter the agent). |
| `dreamerv3-torch/exploration.py` | Exploration behaviors. The configured default is `greedy` (the task behavior); `random` and `plan2explore` exist but are not used by default. |
| `dreamerv3-torch/parallel.py` | Environment process/driver helpers (`Parallel`, `Damy`). For the `turtle` task a single environment is used. |
| `dreamerv3-torch/verify_gpu.py` | Standalone GPU/CUDA sanity check. |

**Excluded as legacy / out of scope:** `model_free/ddpg/`, `model_free/sac/`,
`model_free/td3/`, `model_free/util/`, and the root-level model-free scripts
(`train.py`, `tools.py`, `test.py`, `plt.py`, `save_to_best.py`) and `turtle_env/`.

> Document set generated by static inspection of the source code only. No source
> code was modified; no training run or simulator was launched to produce it.
