#!/usr/bin/env python3
"""scripts/repair_csv_nuls.py — Strip trailing NUL-byte padding from CSV files
left by an ext4 delayed-allocation crash (power loss / hard hang mid-write).

Conservative by design: only removes a PURE run of NUL bytes found strictly after
the last newline in the file. If anything else (a partial/incomplete last row) is
found after the last newline, the file is left untouched and reported for manual
inspection — this tool never rewrites, reflows, or reinterprets CSV content, it
only truncates trailing zero-padding.

Always writes a `<file>.bak` copy before modifying anything. Dry-run by default —
pass --apply to actually modify files.

Usage (dry run — report only, no changes):
  python3 scripts/repair_csv_nuls.py --csv-dir ./csv_logs/full_imu_stage4

Usage (apply fixes):
  python3 scripts/repair_csv_nuls.py --csv-dir ./csv_logs/full_imu_stage4 --apply

Usage (single file):
  python3 scripts/repair_csv_nuls.py --file ./csv_logs/full_imu_stage4/blackbox_run.csv --apply
"""
from __future__ import annotations
import argparse
import pathlib
import shutil


def _inspect(path: pathlib.Path):
    """Return (n_trailing_nuls, has_partial_tail, last_newline_pos)."""
    data = path.read_bytes()
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return 0, False, -1  # no newline at all — nothing this tool handles
    tail = data[last_nl + 1:]
    if len(tail) == 0:
        return 0, False, last_nl
    if all(b == 0 for b in tail):
        return len(tail), False, last_nl
    # Non-NUL bytes after the last newline -> partial/incomplete last row, not
    # pure padding. Not this tool's job to fix.
    return 0, True, last_nl


def _repair_file(path: pathlib.Path, apply: bool) -> str:
    n_nuls, has_partial, last_nl = _inspect(path)
    if last_nl == -1:
        return "skip (no newline found in file)"
    if n_nuls == 0 and not has_partial:
        return "clean (no trailing NULs)"
    if has_partial:
        return ("SKIPPED — trailing bytes after last newline are not pure NUL "
                 "padding (partial/incomplete last row); left untouched, inspect manually")
    if not apply:
        return f"would strip {n_nuls} trailing NUL byte(s) (dry run — pass --apply to fix)"
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with open(path, "r+b") as f:
        f.truncate(last_nl + 1)
    return f"stripped {n_nuls} trailing NUL byte(s) (backup: {backup.name})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv-dir", help="scan every *.csv in this directory")
    g.add_argument("--file", help="repair a single CSV file")
    ap.add_argument("--apply", action="store_true",
                    help="actually modify files (creates a .bak first). Without this, "
                         "the script only reports what it would do.")
    args = ap.parse_args()

    if args.file:
        paths = [pathlib.Path(args.file)]
    else:
        paths = sorted(pathlib.Path(args.csv_dir).glob("*.csv"))
        if not paths:
            print(f"No .csv files found in {args.csv_dir}")
            return

    for p in paths:
        if not p.exists():
            print(f"{p}: NOT FOUND")
            continue
        result = _repair_file(p, args.apply)
        print(f"{p.name}: {result}")

    if not args.apply:
        print("\n[info] dry run only — no files modified. Re-run with --apply to fix.")


if __name__ == "__main__":
    main()
