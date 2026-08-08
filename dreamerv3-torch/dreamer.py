import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    """Sum validated step counts across saved episodes in `folder`. Opens each
    .npz and checks for a 'reward' key (tools.validate_episode_file) rather than
    trusting the filename-encoded length blindly, so a corrupt/truncated file is
    excluded instead of silently inflating the count. Returns (total_steps, n_skipped).
    """
    folder = pathlib.Path(folder)
    total = 0
    skipped = []
    for n in sorted(folder.glob("*.npz")):
        ok, length, err = tools.validate_episode_file(n)
        if ok:
            total += length - 1
        else:
            skipped.append((n.name, err))
    if skipped:
        names = ", ".join(name for name, _ in skipped[:5])
        more = f", … (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
        print(f"[count_steps] Warning: skipped {len(skipped)} corrupt/invalid .npz "
              f"file(s) in {folder} (excluded from step count): {names}{more}")
    return total, len(skipped)


def _resolve_checkpoint(logdir, traindir, resume_policy, confirm_fresh):
    """Resolve which checkpoint (if any) to load at startup, per --resume_policy.
    Never lets a corrupt/missing checkpoint crash with a raw traceback — prints a
    clear message and sys.exit()s for any case that should abort instead of
    silently guessing. Returns (checkpoint_dict_or_None, loaded_path_or_None).
    """
    if resume_policy not in ("auto", "strict", "fresh"):
        sys.exit(f"[resume] Invalid --resume_policy {resume_policy!r} — must be "
                 f"one of 'auto', 'strict', 'fresh'.")

    latest = logdir / "latest.pt"
    backup = logdir / "latest_prev.pt"
    tmp = logdir / "latest.pt.tmp"
    if tmp.exists():
        print(f"[resume] Note: found leftover {tmp.name} — a previous checkpoint "
              f"save was interrupted mid-write. It is not auto-loaded (only "
              f"{latest.name} / {backup.name} are trusted); safe to ignore or delete.")

    def _try(path):
        if not path.exists():
            return None
        try:
            ckpt = torch.load(path)
            if "agent_state_dict" not in ckpt or "optims_state_dict" not in ckpt:
                raise ValueError("checkpoint missing expected keys")
            return ckpt
        except Exception as e:
            print(f"[resume] Could not load {path}: {e}")
            return None

    traindir_path = pathlib.Path(traindir)
    non_empty = latest.exists() or backup.exists() or any(traindir_path.glob("*.npz"))

    if resume_policy == "fresh":
        if non_empty and not confirm_fresh:
            sys.exit(
                f"[resume] --resume_policy fresh was given but {logdir} already contains "
                f"a checkpoint and/or replay episodes. Refusing to silently start fresh — "
                f"that would discard existing progress and confuse experiment interpretation. "
                f"Re-run with --confirm_fresh to proceed anyway, or use a new --logdir, or "
                f"drop --resume_policy fresh to resume normally."
            )
        if non_empty:
            n_leftover = len(list(traindir_path.glob("*.npz")))
            print(f"[resume] --resume_policy fresh --confirm_fresh: starting the AGENT "
                  f"fresh (no checkpoint loaded), but {n_leftover} pre-existing episode "
                  f"file(s) remain in {traindir} and will still be counted/replayed. "
                  f"Use a new --logdir for a fully clean run.")
        return None, None

    ckpt = _try(latest)
    if ckpt is not None:
        return ckpt, latest

    if resume_policy == "strict":
        if latest.exists():
            sys.exit(
                f"[resume] --resume_policy strict: {latest} exists but failed to load "
                f"(see error above), and strict mode does not fall back to a backup. "
                f"Re-run with --resume_policy auto to allow falling back to {backup.name}, "
                f"or --resume_policy fresh --confirm_fresh to discard this run's checkpoint."
            )
        return None, None  # no latest.pt at all -> genuinely fresh, nothing to be strict about

    # auto (default)
    ckpt = _try(backup)
    if ckpt is not None:
        print(f"[resume] Recovered using backup checkpoint {backup} "
              f"(primary {latest.name} was missing or failed to load).")
        return ckpt, backup

    if latest.exists() or backup.exists():
        sys.exit(
            f"[resume] --resume_policy auto: both {latest.name} and {backup.name} in "
            f"{logdir} are missing or failed to load (see errors above). Aborting instead "
            f"of silently starting fresh in a non-empty logdir. Re-run with "
            f"--resume_policy fresh --confirm_fresh if you intend to discard this run's "
            f"checkpoint."
        )
    return None, None  # neither file exists -> genuinely fresh logdir


def _make_checkpoint_fn(agent, logdir):
    """Returns a zero-arg closure that atomically saves latest.pt (with backup
    rotation), reading the live agent/optimizer state at call time. Passed to
    tools.simulate() as checkpoint_fn — see its docstring for the non-invasive
    hook contract (no env/RSSM/episode side effects, purely an extra save)."""
    def _fn():
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        tools.atomic_torch_save(items_to_save, logdir / "latest.pt", keep_backup=True)
        print(f"[checkpoint] saved latest.pt at step {agent._step}")
    return _fn


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    import pathlib
    import envs.turtle as turtle
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    run_name = pathlib.Path(config.logdir).name if config.logdir else 'baseline'
    env = turtle.Turtle(
        config.stage,
        config.time_limit,
        config.lidar,
        run_name=run_name,
        mode=mode,
        odometry_mode=config.odometry_mode,
        device=config.device,
        resource_logging=config.resource_logging,
        reward_mode=config.reward_mode,
        reward_progress_scale=config.reward_progress_scale,
        reward_step_penalty=config.reward_step_penalty,
        reward_turn_penalty=config.reward_turn_penalty,
        reward_near_obstacle_scale=config.reward_near_obstacle_scale,
        reward_near_obstacle_sigma=config.reward_near_obstacle_sigma,
        pbrs_scale=config.pbrs_scale,
        pbrs_distance_weight=config.pbrs_distance_weight,
        pbrs_angle_weight=config.pbrs_angle_weight,
        pbrs_distance_scale=config.pbrs_distance_scale,
        pbrs_gamma=config.pbrs_gamma,
        fixed_goals=config.fixed_goals,
        fixed_goals_random=config.fixed_goals_random,
        max_linear_vel=config.max_linear_vel,
        max_angular_vel=config.max_angular_vel,
        csv_dir=config.csv_dir,
        plots_dir=config.plots_dir,
    )
    env = wrappers.UUID(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    # Exploration override (Part C / BO lever): when --actor_entropy >= 0, override the
    # nested actor.entropy coefficient before the agent is built. Sentinel -1.0 = leave
    # the config default (3e-4) untouched. config.actor is a mutable dict on the namespace.
    if getattr(config, "actor_entropy", -1.0) >= 0:
        config.actor["entropy"] = config.actor_entropy
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # Auto-organize every run into a per-mode-per-stage subfolder
    # ({odometry_mode}_stage{N}, e.g. none_stage1/, full_stage1/,
    # full_imu_stage1/) — mirrors BO's csv_logs/tune_stage{N}/ so runs never
    # scatter flat in csv_logs/. Mutating config here covers both the whitebox
    # Logger (below) and make_env (blackbox/planning/reward/resource/plots).
    # Skipped if the user overrode --csv_dir / --plots_dir (e.g. BO trials,
    # which set their own --csv_dir).
    _run_subdir = f'{config.odometry_mode}_stage{config.stage}'
    if config.csv_dir == './csv_logs':
        config.csv_dir = f'./csv_logs/{_run_subdir}'
    if config.plots_dir == './path_plots':
        config.plots_dir = f'./path_plots/{_run_subdir}'

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step, n_skipped_initial = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step, log_videos=config.log_videos, csv_dir=config.csv_dir)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - step)
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    # eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    checkpoint, loaded_ckpt_path = _resolve_checkpoint(
        logdir, config.traindir, config.resume_policy, config.confirm_fresh)
    if checkpoint is not None:
        print(f"[resume] Loaded checkpoint: {loaded_ckpt_path}")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    else:
        print("[resume] Starting fresh — no checkpoint loaded.")

    # Resume event log — one row per process start, so a crash/resume history is
    # visible without cross-referencing timestamps across the other CSVs. Lives in
    # logdir/ (gitignored), not csv_dir, since it's a per-run breadcrumb, not a metric.
    try:
        import csv as _csv
        import datetime as _dt
        _re_path = logdir / "resume_events.csv"
        _re_is_new = not _re_path.exists()
        with open(_re_path, "a", newline="") as _re_f:
            _re_w = _csv.writer(_re_f)
            if _re_is_new:
                _re_w.writerow(["datetime", "resume_policy", "checkpoint_loaded", "step",
                                "replay_episodes_loaded", "corrupt_episodes_skipped"])
            _re_w.writerow([
                _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                config.resume_policy,
                str(loaded_ckpt_path) if loaded_ckpt_path else "none (fresh start)",
                step,
                len(train_eps),
                n_skipped_initial,
            ])
    except Exception as e:
        print(f"[resume] Warning: could not write resume_events.csv: {e}")
    print(f"[resume] step={step}  replay_episodes={len(train_eps)}  "
          f"corrupt_skipped={n_skipped_initial}  resume_policy={config.resume_policy}")

    # Loop bound is steps + eval_every so the model AT config.steps gets evaluated
    # (eval runs at the top of each iteration); the break below then stops before
    # training a wasted post-final-eval chunk (saves ~eval_every steps per run).
    ctr = -1
    best_eval = -np.inf
    while agent._step < config.steps + config.eval_every:
        ctr += 1
        logger.write()
        if config.eval_episode_num > 0:  # evaluate every eval_every steps (one training round per iteration)
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            eval_ret = tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if np.array(eval_ret).mean() >= best_eval:
                best_eval = np.array(eval_ret).mean()
                print('New best mean: ', best_eval)
                items_to_save = {
                            "agent_state_dict": agent.state_dict(),
                            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                }
                tools.atomic_torch_save(items_to_save, logdir / "best.pt", keep_backup=False)
                data = pd.DataFrame({'scores': eval_ret})
                data.to_csv(f'./{logdir}/best.csv')

        if agent._step >= config.steps:   # final-budget eval done above — don't train a wasted tail
            break

        print("Start training.")
        _checkpoint_fn = _make_checkpoint_fn(agent, logdir) if config.checkpoint_every > 0 else None
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            checkpoint_every=config.checkpoint_every,
            checkpoint_fn=_checkpoint_fn,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        tools.atomic_torch_save(items_to_save, logdir / "latest.pt", keep_backup=True)
        print(f"[checkpoint] saved latest.pt at step {agent._step}")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    yaml = YAML(typ='safe', pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
