#!/usr/bin/env python3
"""Export an Optuna reward-tuning study to CSV under csv_logs/tune_stage{N}/.

Writes two files next to the per-trial CSVs:
  tune_trials_stage{N}.csv  — every trial (best AND non-best), one row each,
                              with the five reward weights and the trial score.
  tune_best_stage{N}.csv    — the single best *feasible* config (one row).

Read-only on the study DB, so it is safe to run while the study is still
going — just re-run it after all trials finish to refresh the snapshot.

Usage:
  python3 export_tune_results.py --stage 1
  python3 export_tune_results.py --stage 1 --study-name reward_stage1
"""
import argparse
import csv
import pathlib

import optuna

THIS_DIR = pathlib.Path(__file__).resolve().parent

# The five shaped-reward weights searched by tune_reward.py (see its SEARCH_SPACE).
REWARD_KEYS = [
    "reward_progress_scale",
    "reward_step_penalty",
    "reward_turn_penalty",
    "reward_near_obstacle_scale",
    "reward_near_obstacle_sigma",
]

BASE_FIELDS = [
    "trial", "run_name", "state", "efficiency", "success", "feasible",
    "constraint", "partial", "n_eff_rows", "rc",
    "datetime_start", "datetime_complete",
]
FIELDNAMES = BASE_FIELDS + REWARD_KEYS


def _fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def _row_for(trial, study_name):
    ua = trial.user_attrs
    constraint = ua.get("constraint", (None,))
    c = constraint[0] if isinstance(constraint, (list, tuple)) else constraint
    feasible = c is not None and c <= 0 and trial.value is not None
    row = {
        "trial": trial.number,
        "run_name": f"{study_name}_trial{trial.number:03d}",
        "state": trial.state.name,
        "efficiency": trial.value,
        "success": ua.get("success"),
        "feasible": feasible,
        "constraint": c,
        "partial": ua.get("partial"),
        "n_eff_rows": ua.get("n_eff_rows"),
        "rc": ua.get("rc"),
        "datetime_start": _fmt_dt(trial.datetime_start),
        "datetime_complete": _fmt_dt(trial.datetime_complete),
    }
    for k in REWARD_KEYS:
        row[k] = trial.params.get(k)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--odometry-mode", default="none",
                    help="match the tune_reward.py run's mode (default none); non-none "
                         "uses the mode-namespaced study/db/csv (e.g. tune_stage{N}_full_imu)")
    ap.add_argument("--study-name", default=None,
                    help="default: reward_stage{stage}[_{mode}]")
    ap.add_argument("--storage", default=None,
                    help="default: sqlite:///tune_reward_stage{stage}[_{mode}].db in this dir")
    ap.add_argument("--csv-dir", default=None,
                    help="default: ./csv_logs/tune_stage{stage}[_{mode}]")
    args = ap.parse_args()

    # Mirror tune_reward.py's mode namespacing (empty suffix for none).
    mode_suffix = "" if args.odometry_mode == "none" else f"_{args.odometry_mode}"
    study_name = args.study_name or f"reward_stage{args.stage}{mode_suffix}"
    storage = args.storage or \
        f"sqlite:///{THIS_DIR / f'tune_reward_stage{args.stage}{mode_suffix}.db'}"
    csv_dir = pathlib.Path(
        args.csv_dir or (THIS_DIR / "csv_logs" / f"tune_stage{args.stage}{mode_suffix}"))
    csv_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.load_study(study_name=study_name, storage=storage)
    baseline_success = study.user_attrs.get("baseline_success")
    margin = study.user_attrs.get("margin")

    rows = [_row_for(t, study_name) for t in study.trials]

    # Sort: feasible first, then by efficiency descending (None last).
    def sort_key(r):
        eff = r["efficiency"] if r["efficiency"] is not None else float("-inf")
        return (0 if r["feasible"] else 1, -eff)
    rows.sort(key=sort_key)

    trials_path = csv_dir / f"tune_trials_stage{args.stage}.csv"
    with open(trials_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    completed = [r for r in rows if r["efficiency"] is not None]
    feasible = [r for r in completed if r["feasible"]]
    best = feasible[0] if feasible else (completed[0] if completed else None)

    best_path = csv_dir / f"tune_best_stage{args.stage}.csv"
    with open(best_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        if best is not None:
            w.writerow(best)

    # ── Console summary ──────────────────────────────────────────────────────
    n_done = len(completed)
    n_feas = len(feasible)
    print(f"[export] study '{study_name}'  "
          f"trials={len(study.trials)}  completed={n_done}  feasible={n_feas}")
    print(f"[export] baseline_success={baseline_success}  margin={margin}")
    print(f"[export] all trials -> {trials_path}")
    print(f"[export] best       -> {best_path}")
    if best is None:
        print("[export] no completed trials yet.")
        return
    tag = "feasible" if best["feasible"] else "NO feasible trial — best completed"
    print(f"\n========== BEST ({tag}) ==========")
    print(f"trial {best['trial']}  run_name={best['run_name']}  "
          f"efficiency={best['efficiency']:.4f}  success={best['success']}")
    print("validation flags:")
    print("  " + " ".join(f"--{k} {best[k]:.6g}" for k in REWARD_KEYS))


if __name__ == "__main__":
    main()
