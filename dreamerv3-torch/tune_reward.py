#!/usr/bin/env python3
"""Bayesian optimization of reward-shaping weights for the shaped reward mode.

This is a STANDALONE orchestration script. It does NOT modify the reward/observation
implementation, the odometry-mode implementation, the DreamerV3 architecture, the A*
metric, the dashboard, or torch/CUDA. Each Optuna trial simply launches ``dreamer.py``
as a subprocess with a candidate set of ``--reward_*`` flags, waits for training to
finish, and scores the run by parsing the eval CSVs it produced.

The observation space the weights are tuned under is selectable via ``--odometry-mode``
(default ``none``). Because the obs space materially changes what the policy can learn,
reward weights tuned under ``none`` may not be optimal under ``full_imu``; tune under the
mode you will deploy. The mode always namespaces the study/db/CSV/plots folders
(e.g. ``tune_stage1_none``, ``tune_stage1_full_imu``), so studies for the same stage but
different modes never collide.

Objective (constrained efficiency):
    maximize  eval planner_path_efficiency
    subject to  eval success_rate >= baseline_success - margin

A* is used here only as the *scoring metric* (read from planning_eval_<run>.csv).
It is never fed into training.

Design notes:
  * Gazebo must already be running for the chosen --stage (it stays up across all
    trials since the stage is fixed). Start it once, manually, before running this.
  * Trials run strictly sequentially (one Gazebo, one GPU).
  * The study is persisted to a sqlite DB with load_if_exists=True, so a crash or
    reboot resumes the study where it left off.

Typical use (use --steps >=80000 so SCORE_WINDOW=60 lands on the last 3 trained
evals and skips the untrained ctr=0 eval; see SCORE_WINDOW below):
    # 1) establish the default-reward baseline (constraint floor) once
    python3 tune_reward.py --stage 1 --run-baseline --steps 80000 --eval-episode-num 20
    # 2) run the search
    python3 tune_reward.py --stage 1 --n-trials 30 --steps 80000 --eval-episode-num 20
"""

import argparse
import csv
import math
import os
import pathlib
import subprocess
import sys
import time

THIS_DIR = pathlib.Path(__file__).resolve().parent

# Search space for the 5 shaped-reward weights. Progress is capped low on purpose:
# an over-strong progress reward straight-lines on the near-empty stage 1 and then
# fails to generalize to cluttered stages (see plan "Risks").
SEARCH_SPACE = {
    "reward_progress_scale":      ("log",     0.1, 3.0),
    "reward_step_penalty":        ("uniform", 0.0, 0.05),
    "reward_turn_penalty":        ("uniform", 0.0, 0.05),
    "reward_near_obstacle_scale": ("uniform", 0.0, 0.5),
    "reward_near_obstacle_sigma": ("uniform", 0.1, 0.5),
}

# Number of trailing eval rows (episodes) used to score a run — end-of-training
# behavior. 60 = the last 3 evals at the default eval_every=20000 + --eval-episode-num
# 20, i.e. the 40k/60k/80k checkpoints of an **80k** trial, which excludes the
# untrained ctr=0 eval. IMPORTANT: this assumes BO trials run ≥80k steps (≥5 evals,
# 100 rows). A shorter trial (e.g. 40k → 3 evals, 60 rows) has only 60 rows total,
# so a 60-row window would reach back into the untrained ctr=0 eval and pollute the
# score — use --steps ≥80000 for BO, or lower SCORE_WINDOW for short trials. If you
# raise --eval-episode-num, raise this in proportion (≈ N_eval × trained_evals).
SCORE_WINDOW = 60

# Sentinel objective for a failed/crashed/missing run so BO steers away from it.
FAILED_SCORE = 0.0


# ── CSV scoring ──────────────────────────────────────────────────────────────

def _read_rows(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _tail(path, n=25):
    """Echo the last n lines of a file (best-effort) so a hang/crash is visible."""
    try:
        lines = open(path).readlines()
        if lines:
            print(f"[tune] --- last {min(n, len(lines))} lines of {path} ---", flush=True)
            for ln in lines[-n:]:
                print("    " + ln.rstrip(), flush=True)
    except Exception:
        pass


def score_run(run_name, csv_dir, window=SCORE_WINDOW):
    """Return (efficiency, success_fraction, n_eff_rows) from the eval CSVs.

    efficiency      : mean planner_path_efficiency over the last `window` rows whose
                      planner_status == 'ok'  (0.0 if none).
    success_fraction: fraction of the last `window` eval episodes that succeeded
                      (None if the blackbox eval CSV is missing/empty).
    """
    pl_rows = _read_rows(os.path.join(csv_dir, f"planning_eval_{run_name}.csv"))
    ok = [r for r in pl_rows if r.get("planner_status") == "ok"]
    ok = ok[-window:]
    effs = []
    for r in ok:
        try:
            effs.append(float(r["planner_path_efficiency"]))
        except (KeyError, ValueError, TypeError):
            continue
    efficiency = sum(effs) / len(effs) if effs else 0.0

    bb_rows = _read_rows(os.path.join(csv_dir, f"blackbox_eval_{run_name}.csv"))
    bb = bb_rows[-window:]
    if bb:
        n_succ = sum(1 for r in bb if r.get("outcome") == "success")
        success_fraction = 100.0 * n_succ / len(bb)
    else:
        success_fraction = None

    return efficiency, success_fraction, len(effs)


# ── Subprocess launch ────────────────────────────────────────────────────────

def build_command(args, logdir, reward_mode, weights):
    cmd = [
        sys.executable, "dreamer.py",
        "--configs", "turtle", "--task", "turtle",
        "--logdir", str(logdir),
        "--stage", str(args.stage),
        "--lidar", "360",
        "--odometry_mode", args.odometry_mode,
        "--seed", str(args.seed),
        "--device", args.device,
        "--steps", str(args.steps),
        "--eval_episode_num", str(args.eval_episode_num),
        "--reward_mode", reward_mode,
        "--csv_dir", args.csv_dir,     # trial CSVs land in this (per-study) subfolder
        "--plots_dir", args.plots_dir, # trial PNGs land in this (per-study) subfolder
    ]
    if getattr(args, "eval_every", None) is not None:
        cmd += ["--eval_every", str(args.eval_every)]
    for key, val in weights.items():
        cmd += [f"--{key}", f"{val:.6g}"]
    return cmd


def run_training(args, logdir, reward_mode, weights):
    """Launch dreamer.py; return its return code (0 = success).

    dreamer.py's stdout+stderr are captured to ``{logdir}/dreamer.log`` so a hang
    or crash leaves a post-mortem artifact (previously the output was lost to the
    parent terminal). On timeout / non-zero exit the log tail is echoed.
    """
    cmd = build_command(args, logdir, reward_mode, weights)
    print("[tune] launching:", " ".join(cmd), flush=True)
    if args.dry_run:
        print("[tune] --dry-run: not executing.", flush=True)
        return 0
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "dreamer.log"
    print(f"[tune] dreamer.py output -> {log_path}", flush=True)
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, cwd=str(THIS_DIR), timeout=args.timeout_per_trial,
                stdout=logf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            print(f"[tune] dreamer.py exited rc={proc.returncode} (see {log_path})",
                  flush=True)
            _tail(log_path)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[tune] TIMEOUT after {args.timeout_per_trial}s (see {log_path})",
              flush=True)
        _tail(log_path)
        return 124


# ── Gazebo pre-check ─────────────────────────────────────────────────────────

def gazebo_ready(retries=3, delay=2.0):
    """True if /reset_simulation is advertised (i.e. a Gazebo stage is running).

    Retries a few times: the `ros2 service list` CLI relies on a background
    discovery daemon that can briefly cache an empty/stale graph, so a single
    miss does not mean the sim is down. Without retries a transient hiccup in
    the per-trial check would needlessly stop the whole study.
    """
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["ros2", "service", "list"], capture_output=True, text=True, timeout=15
            )
            if "/reset_simulation" in out.stdout:
                return True
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return False


# ── Baseline ─────────────────────────────────────────────────────────────────

def baseline_run_name(study_name, seed):
    return f"{study_name}_baseline_seed{seed}"


def run_baseline(args, csv_dir, study_name):
    """Run one default-reward training to fix the constraint floor.

    Idempotent: if a baseline with usable eval data already exists, reuse its
    score instead of re-training (avoids silently appending to / continuing the
    baseline on a re-run). A timed-out baseline is still scored on partial data.
    """
    run_name = baseline_run_name(study_name, args.seed)
    if not args.dry_run:
        eff0, succ0, n0 = score_run(run_name, csv_dir)
        if n0 > 0:
            print(f"[tune] baseline already present: success={succ0} "
                  f"efficiency={eff0:.4f} (n={n0}) — reusing, not re-running.",
                  flush=True)
            return eff0, succ0
    logdir = pathlib.Path(args.logdir_root) / study_name / run_name
    rc = run_training(args, logdir, "default", {})  # default mode = no shaping
    if args.dry_run:
        return None
    eff, succ, n = score_run(run_name, csv_dir)
    if n == 0:
        print(f"[tune] baseline produced no eval data (rc={rc}).", flush=True)
        return None
    if rc != 0:
        print(f"[tune] baseline stopped early (rc={rc}); scoring partial data.",
              flush=True)
    print(f"[tune] baseline: success={succ} efficiency={eff:.4f} (n={n})", flush=True)
    return eff, succ


# ── Optuna study ─────────────────────────────────────────────────────────────

def suggest_weights(trial):
    weights = {}
    for key, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "log":
            weights[key] = trial.suggest_float(key, lo, hi, log=True)
        else:
            weights[key] = trial.suggest_float(key, lo, hi)
    return weights


def make_objective(args, csv_dir, baseline_success, study_name):
    import optuna

    def objective(trial):
        # Re-check the sim before every trial: a mid-study Gazebo/DDS-discovery
        # failure would otherwise let each trial hang for the full --timeout-per-trial
        # producing no data. gazebo_ready() runs `ros2 service list`, so it reports
        # not-ready exactly when /reset_simulation is undiscoverable. Stop cleanly
        # (the study is resumable) instead of burning the timeout N times.
        if not args.dry_run and not gazebo_ready():
            print("[tune] ERROR: Gazebo/reset service not discoverable. Stopping study "
                  f"so trials don't burn the {args.timeout_per_trial}s timeout. Fix the "
                  "sim (e.g. ROS_LOCALHOST_ONLY=1, relaunch Gazebo) and re-run to resume.",
                  flush=True)
            trial.study.stop()
            raise optuna.TrialPruned()

        weights = suggest_weights(trial)
        run_name = f"{study_name}_trial{trial.number:03d}"
        logdir = pathlib.Path(args.logdir_root) / study_name / run_name

        rc = run_training(args, logdir, "shaped", weights)
        if args.dry_run:
            # Report a constraint and a placeholder score without touching CSVs.
            trial.set_user_attr("constraint", (0.0,))
            trial.set_user_attr("efficiency", 0.0)
            trial.set_user_attr("success", None)
            return FAILED_SCORE

        # Always try to score from the eval CSVs — a timed-out / non-zero-exit run
        # still produced partial eval episodes that are a valid (if shorter) proxy.
        # Only truly empty runs (crashed before the first eval) are marked failed.
        eff, succ, n = score_run(run_name, csv_dir)
        if n == 0:
            trial.set_user_attr("constraint", (1.0,))  # infeasible
            trial.set_user_attr("efficiency", FAILED_SCORE)
            trial.set_user_attr("success", None)
            trial.set_user_attr("rc", rc)
            print(f"[tune] trial {trial.number}: NO eval data (rc={rc}) "
                  f"-> {FAILED_SCORE}", flush=True)
            return FAILED_SCORE

        partial = (rc != 0)  # timed out or non-zero exit, but partial data exists
        # Constraint c <= 0 is feasible.  c = (floor) - success.
        floor = (baseline_success - args.margin) if baseline_success is not None else 0.0
        succ_val = succ if succ is not None else 0.0
        c = floor - succ_val
        trial.set_user_attr("constraint", (c,))
        trial.set_user_attr("efficiency", eff)
        trial.set_user_attr("success", succ)
        trial.set_user_attr("n_eff_rows", n)
        trial.set_user_attr("partial", partial)
        trial.set_user_attr("rc", rc)
        tag = f" [PARTIAL rc={rc}]" if partial else ""
        print(f"[tune] trial {trial.number}: success={succ} "
              f"efficiency={eff:.4f} constraint={c:.2f} "
              f"({'feasible' if c <= 0 else 'INFEASIBLE'}){tag}", flush=True)
        return eff

    return objective


def constraints_func(trial):
    # Optuna reads the per-trial constraint from this user attr.
    return trial.user_attrs.get("constraint", (0.0,))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--odometry-mode", default="none",
                   help="obs space to tune the reward weights under (none/twist/delta/"
                        "full/full_imu; default none). The mode always namespaces the "
                        "study/db/CSV/plots (e.g. tune_stage{N}_none, tune_stage{N}_full_imu) "
                        "so studies don't collide. Forwarded to dreamer.py as --odometry_mode.")
    p.add_argument("--steps", type=int, default=80000,
                   help="proxy training budget per trial (>=80000 so SCORE_WINDOW=60 "
                        "lands on the last 3 trained evals, skipping the untrained eval)")
    p.add_argument("--eval-episode-num", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=None,
                   help="eval interval in steps forwarded to dreamer.py. Default: "
                        "omit it -> dreamer.py uses its config default (20000). "
                        "Lower it (e.g. 2000) for a fast smoke test.")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--margin", type=float, default=5.0,
                   help="allowed success-rate drop vs baseline, in percentage points")
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (default: sqlite in this dir)")
    p.add_argument("--logdir-root", default="./logdir")
    p.add_argument("--csv-dir", default=None,
                   help="base dir for trial CSVs (default: ./csv_logs/tune_stage{N}, "
                        "so the ~9 CSVs/trial stay out of the main csv_logs/)")
    p.add_argument("--plots-dir", default=None,
                   help="base dir for path-plot PNGs (default: ./path_plots/tune_stage{N}, "
                        "mirrors --csv-dir so BO plots stay separate from main path_plots/)")
    p.add_argument("--timeout-per-trial", type=int, default=86400,
                   help="seconds before a trial is stopped; partial eval data is "
                        "still scored. Default is 24 h so a trial is never cut off "
                        "unnoticed (40k steps need ~140 min; larger --steps need "
                        "more). Lower it only if you want a hard per-trial cap.")
    p.add_argument("--baseline-success", type=float, default=None,
                   help="skip the baseline run and use this success-rate floor directly")
    p.add_argument("--run-baseline", action="store_true",
                   help="run one default-reward baseline, print its metrics, and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print the dreamer.py commands without launching anything")
    args = p.parse_args()

    # Namespace the study/db/CSV/plots by odometry mode so tuning under different
    # obs spaces (e.g. none vs full_imu) never share a study or folder. The mode
    # is always part of the name (incl. 'none' -> _none) for consistent naming.
    mode_suffix = f"_{args.odometry_mode}"
    study_name = args.study_name or f"reward_stage{args.stage}{mode_suffix}"
    storage = args.storage or f"sqlite:///{THIS_DIR / f'tune_reward_stage{args.stage}{mode_suffix}.db'}"
    if args.csv_dir is None:
        args.csv_dir = f"./csv_logs/tune_stage{args.stage}{mode_suffix}"  # contain the trial CSVs
    if args.plots_dir is None:
        args.plots_dir = f"./path_plots/tune_stage{args.stage}{mode_suffix}"  # mirror csv_dir structure
    # absolute path used for reading/scoring; dreamer.py writes to the same (cwd=THIS_DIR)
    csv_dir = args.csv_dir if os.path.isabs(args.csv_dir) else os.path.join(THIS_DIR, args.csv_dir)

    # Gazebo pre-check (skipped on dry-run).
    if not args.dry_run and not gazebo_ready():
        print("[tune] ERROR: /reset_simulation not found — start Gazebo for "
              f"stage {args.stage} first:\n"
              "  export TURTLEBOT3_MODEL=burger\n"
              f"  ros2 launch ~/turtlebot-dreamerv3/turtlebot3_gazebo/launch/"
              f"turtle_stage{args.stage}.py", flush=True)
        sys.exit(2)

    if args.run_baseline:
        run_baseline(args, csv_dir, study_name)
        return

    # Resolve the constraint floor.
    baseline_success = args.baseline_success
    if baseline_success is None and not args.dry_run:
        print("[tune] no --baseline-success given; running a baseline first...", flush=True)
        res = run_baseline(args, csv_dir, study_name)
        baseline_success = res[1] if res else None
    if baseline_success is None and not args.dry_run:
        print("[tune] WARNING: no baseline success available; constraint floor = 0 "
              "(efficiency maximized unconstrained).", flush=True)

    import optuna
    sampler = optuna.samplers.TPESampler(constraints_func=constraints_func, seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    if baseline_success is not None:
        study.set_user_attr("baseline_success", baseline_success)
        study.set_user_attr("margin", args.margin)

    def _export_csvs(study_=None, trial_=None):
        # Keep tune_trials_stage{N}.csv and tune_best_stage{N}.csv fresh after
        # every trial and at study end (best-effort; non-fatal to the study).
        if args.dry_run:
            return
        try:
            subprocess.run(
                [sys.executable, str(THIS_DIR / "export_tune_results.py"),
                 "--stage", str(args.stage), "--study-name", study_name,
                 "--storage", storage, "--csv-dir", str(csv_dir)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

    study.optimize(make_objective(args, csv_dir, baseline_success, study_name),
                   n_trials=args.n_trials, callbacks=[_export_csvs])

    if args.dry_run:
        print("[tune] dry-run complete.", flush=True)
        return

    _export_csvs()  # final refresh after the last trial

    # Report the best *feasible* trial if one exists, else the best completed.
    # Use only COMPLETE trials (value is not None) — a study stopped on a Gazebo
    # outage may hold only PRUNED trials, and study.best_trial would raise.
    completed = [t for t in study.trials if t.value is not None]
    feasible = [t for t in completed
                if t.user_attrs.get("constraint", (1.0,))[0] <= 0]
    best = (max(feasible, key=lambda t: t.value) if feasible
            else (max(completed, key=lambda t: t.value) if completed else None))
    print("\n========== BEST CONFIG ==========", flush=True)
    if best is None:
        print("no completed trials.", flush=True)
        return
    print(f"trial {best.number}  efficiency={best.value:.4f}  "
          f"success={best.user_attrs.get('success')}  "
          f"feasible={best in feasible}", flush=True)
    for k, v in best.params.items():
        print(f"  --{k} {v:.6g}", flush=True)
    print("\nNext: validate this config at full budget across 3 seeds on stage 1, "
          "then on stages 2-4 to test transfer (see plan / CLAUDE.md).", flush=True)


if __name__ == "__main__":
    main()
