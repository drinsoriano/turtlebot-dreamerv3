#!/usr/bin/env python3
"""scripts/scan_corrupt_episodes.py — Scan a logdir's train_eps/ and eval_eps/
folders for corrupt/unreadable .npz episode files.

Read-only by default: reports corrupt files and their (best-effort, filename-
derived) affected step counts. Does NOT delete anything unless --clean_corrupt_eps
is explicitly passed, in which case it prints exactly which files it removes
before removing them.

Deliberately dependency-light (numpy + stdlib only, no torch/tools.py import) so
it stays fast for repeated interactive use right after a crash. Duplicates the
validation check in tools.validate_episode_file() — keep both in sync.

Usage:
  python3 scripts/scan_corrupt_episodes.py --logdir ./logdir/stage4_..._v2
  python3 scripts/scan_corrupt_episodes.py --logdir ./logdir/stage4_..._v2 --clean_corrupt_eps
"""
from __future__ import annotations
import argparse
import pathlib

import numpy as np


def _validate(path: pathlib.Path):
    """Mirrors tools.validate_episode_file — keep in sync if criteria change."""
    try:
        if path.stat().st_size == 0:
            return False, "0-byte file"
        with path.open("rb") as f:
            with np.load(f) as npz:
                if "reward" not in npz.files:
                    return False, "missing 'reward' key"
        return True, None
    except Exception as e:
        return False, str(e)


def _scan_dir(d: pathlib.Path):
    corrupt = []
    total_files = 0
    if not d.exists():
        return corrupt, total_files
    for f in sorted(d.glob("*.npz")):
        total_files += 1
        ok, err = _validate(f)
        if not ok:
            # Best-effort estimate from the filename convention {name}-{length}.npz —
            # purely informational, since the file couldn't be confirmed to actually
            # contain this many steps.
            est = None
            try:
                est = int(f.name.split("-")[-1][:-4]) - 1
            except Exception:
                pass
            corrupt.append((f, err, est))
    return corrupt, total_files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True, help="run logdir, e.g. ./logdir/stage4_..._v2")
    ap.add_argument("--clean_corrupt_eps", action="store_true",
                    help="delete the corrupt files found (prints exactly which, before "
                         "deleting). Off by default — this script is read-only unless "
                         "this flag is passed.")
    args = ap.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    train_dir = logdir / "train_eps"
    eval_dir = logdir / "eval_eps"

    grand_total_files = 0
    grand_corrupt = []
    for label, d in (("train_eps", train_dir), ("eval_eps", eval_dir)):
        corrupt, total_files = _scan_dir(d)
        grand_total_files += total_files
        print(f"\n[{label}] {d}")
        print(f"[{label}] scanned {total_files} .npz file(s)")
        if not corrupt:
            print(f"[{label}] no corrupt files found.")
        else:
            print(f"[{label}] {len(corrupt)} corrupt file(s):")
            for f, err, est in corrupt:
                est_str = (f"~{est} steps (filename estimate, unverified)"
                           if est is not None else "steps unknown")
                print(f"  {f.name}  —  {err}  —  {est_str}")
            grand_corrupt.extend(corrupt)

    est_total = sum(e for _, _, e in grand_corrupt if e is not None)
    print(f"\n[summary] {len(grand_corrupt)} corrupt file(s) out of {grand_total_files} "
          f"scanned across train_eps/ + eval_eps/.")
    if grand_corrupt:
        print(f"[summary] estimated {est_total} affected step(s) (filename-derived, "
              f"unverified since the files could not be opened).")

    if args.clean_corrupt_eps:
        if not grand_corrupt:
            print("\n[clean] nothing to remove.")
            return
        print(f"\n[clean] removing {len(grand_corrupt)} corrupt file(s):")
        for f, err, est in grand_corrupt:
            print(f"  removing {f}")
            f.unlink()
        print(f"[clean] done — {len(grand_corrupt)} file(s) removed.")
    else:
        print("\n[info] no files were deleted (read-only by default). "
              "Pass --clean_corrupt_eps to remove the corrupt files listed above.")


if __name__ == "__main__":
    main()
