#!/usr/bin/env python3
"""
preview_fixed_goals.py — Visualize where a --fixed_goals set lands on an arena.

Standalone, read-only: NO Gazebo, NO GPU, NO training. Renders one PNG showing the
stage arena + obstacles, the robot start at the origin (0,0), and every fixed goal
(numbered in round-robin order) with its (x, y) label, the 0.4 m acceptance circle,
and the A* shortest path from the origin. A goal A* cannot reach is drawn red and
labelled "NO PATH" — so this preview doubles as the goal-validity check you run
*before* training.

Reuses the exact training-plot renderer (envs/path_viz.py) and A* planner
(envs/stage_map.py), so the preview matches the per-episode path plots.

Usage:
  python3 preview_fixed_goals.py --stage 4 \
    --fixed_goals "2.0,2.0;-2.0,-2.0;2.0,-2.0;-2.0,2.0"
  # → path_plots/fixed_goals_preview_stage4.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # non-interactive; safe with no display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Make `envs` importable whether run from the repo root or dreamerv3-torch/.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from envs.path_viz import _draw_stage, _draw_robot, REACH_THRESHOLD
from envs.stage_map import (STAGE_ARENAS, get_grid, astar_plan, dubins_plan,
                            RESOLUTION)


def parse_fixed_goals(s: str) -> list[tuple[float, float]]:
    """Parse a ';'-separated 'x,y' list. Mirrors the parser in envs/turtle.py
    (init_properties) exactly so the preview and training never disagree."""
    return [
        (float(p.split(',')[0]), float(p.split(',')[1]))
        for p in s.split(';') if p.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=int, required=True, help='Arena stage (1-8).')
    ap.add_argument('--fixed_goals', type=str, required=True,
                    help="';'-separated 'x,y' list, e.g. '2.0,2.0;-2.0,-2.0'.")
    ap.add_argument('--out', type=str, default='',
                    help='Output PNG path (default path_plots/fixed_goals_preview_stage{N}.png).')
    ap.add_argument('--radius', type=float, default=0.5,
                    help='Min turning radius (m) for the Dubins reference = '
                         'max_linear_vel/max_angular_vel (default 0.5 = 0.1/0.2).')
    ap.add_argument('--start_heading', type=float, default=0.0,
                    help='Robot start heading in radians (default 0 = facing +x).')
    args = ap.parse_args()

    if args.stage not in STAGE_ARENAS:
        print(f"ERROR: stage {args.stage} has no arena geometry (supported: "
              f"{sorted(STAGE_ARENAS)}).")
        return 2

    try:
        goals = parse_fixed_goals(args.fixed_goals)
    except Exception as e:
        print(f"ERROR: could not parse --fixed_goals '{args.fixed_goals}': {e}")
        return 2
    if not goals:
        print("ERROR: --fixed_goals parsed to an empty list.")
        return 2

    grid, arena = get_grid(args.stage)
    start = (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_stage(ax, args.stage)

    # Robot start marker at origin (same green Burger as the training plots),
    # nose drawn at the true start heading the Dubins arc departs from.
    _draw_robot(ax, 0.0, 0.0, heading=args.start_heading,
                body_color='#2ca02c', label='Start (0, 0)')
    start_xyt = (0.0, 0.0, args.start_heading)

    # Colours: Dubins (blue solid) is the primary nonholonomic reference; the old
    # holonomic A* drops to teal dash-dot for contrast (matches the per-episode PNG).
    DUBINS_C, ASTAR_C = '#1f77b4', '#17becf'

    n_ok = 0
    summary: list[str] = []
    for i, goal in enumerate(goals):
        # A* region path (holonomic, stops at the 0.4 m acceptance radius)
        a_len, a_wp = astar_plan(grid, arena, start, goal, REACH_THRESHOLD)
        a_ok = a_len is not None and a_wp
        # Dubins-to-point reference (nonholonomic, respects --radius)
        d_len, d_wp, d_status = dubins_plan(grid, arena, start_xyt, goal,
                                            args.radius, REACH_THRESHOLD)
        d_ok = d_status == 'ok' and d_wp

        ec = '#2ca02c' if (a_ok or d_ok) else '#d62728'  # circle: green ok / red none

        # Acceptance circle (dashed) + faint fill
        ax.add_patch(mpatches.Circle(goal, REACH_THRESHOLD, fill=False,
                                     linestyle='--', edgecolor=ec, linewidth=1.2,
                                     zorder=3))
        ax.add_patch(mpatches.Circle(goal, REACH_THRESHOLD, facecolor=ec,
                                     alpha=0.08, zorder=1))

        # A* overlay (teal dash-dot, secondary)
        if a_ok:
            ax.plot([p[0] for p in a_wp], [p[1] for p in a_wp], '-.',
                    color=ASTAR_C, linewidth=1.4, alpha=0.9, zorder=4,
                    label='A* (holonomic)' if i == 0 else '_nolegend_')
        # Dubins overlay (blue solid, primary)
        if d_ok:
            ax.plot([p[0] for p in d_wp], [p[1] for p in d_wp], '-',
                    color=DUBINS_C, linewidth=2.0, alpha=0.95, zorder=5,
                    label=f'Dubins (r={args.radius:.2f} m)' if i == 0 else '_nolegend_')
            n_ok += 1

        a_txt = f"A*={a_len:.2f}" if a_ok else "A*=blocked"
        d_txt = f"Dub={d_len:.2f}" if d_ok else f"Dub={d_status}"
        tag = f"#{i} ({goal[0]:+.2f},{goal[1]:+.2f})\n{d_txt} | {a_txt}"
        summary.append(f"  goal #{i} ({goal[0]:+.2f}, {goal[1]:+.2f}): {d_txt} m | {a_txt} m")

        # Goal star + index/coord label
        ax.plot(goal[0], goal[1], '*', color=ec, markersize=15, zorder=6)
        ax.annotate(tag, xy=goal, xytext=(6, 6), textcoords='offset points',
                    fontsize=8, color='#333333', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=ec,
                              alpha=0.85, lw=0.8))

    ax.set_title(
        f'Fixed-goal preview — Stage {args.stage}  '
        f'(Dubins r={args.radius:.2f} m, start heading {args.start_heading:.2f} rad)\n'
        f'{len(goals)} goal(s), {n_ok} Dubins-reachable  (round-robin order: #0, #1, …)',
        fontsize=10,
    )
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    pad = 0.3
    ax.set_xlim(arena['x_min'] - pad, arena['x_max'] + pad)
    ax.set_ylim(arena['y_min'] - pad, arena['y_max'] + pad)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()

    out = args.out or f'./path_plots/fixed_goals_preview_stage{args.stage}.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"Stage {args.stage} — {len(goals)} goal(s), {n_ok} Dubins-reachable "
          f"(radius {args.radius:.2f} m):")
    print('\n'.join(summary))
    print(f"\nSaved: {out}")
    if n_ok < len(goals):
        print("NOTE: some goals are Dubins-blocked (the curvature-bounded curve would "
              "hit an obstacle) — efficiency_dubins is honestly blank there; A* may "
              "still route around. Pick other coords if you want a clean Dubins preview.")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
