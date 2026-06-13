"""
path_viz.py — Episode path visualization for debugging and thesis figures.

Generates a 2-D overhead plot per episode showing:
  - Stage arena outline and physical obstacle geometry (not inflated)
  - A* reference path (blue) with endpoint marker
  - Goal acceptance circle (dashed red, radius = REACH_THRESHOLD)
  - Actual robot trajectory (orange)
  - Start (green circle) and goal (red star)

Saved to: {plots_dir}/{run_name}/ep{episode:05d}_{outcome}.png

Called from turtle.py at episode end inside a try/except — never crashes training.
Do not import this module from training-critical code paths outside that guard.
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; safe with no display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from envs.stage_map import STAGE_ARENAS, STAGE_OBSTACLES

REACH_THRESHOLD = 0.4  # metres — must match turtle.py REACH_TRESHOLD


def _path_len(pts: list[tuple[float, float]]) -> float:
    """Euclidean arc-length of a waypoint or trajectory list."""
    return sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )


def _draw_stage(ax, stage: int) -> None:
    """Draw arena boundary and physical obstacle outlines (raw SDF geometry)."""
    arena = STAGE_ARENAS.get(stage)
    if arena is None:
        return

    xmin, xmax = arena['x_min'], arena['x_max']
    ymin, ymax = arena['y_min'], arena['y_max']
    boundary = mpatches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=1.5, edgecolor='black', facecolor='#f5f5f5', zorder=0,
    )
    ax.add_patch(boundary)

    for obs in STAGE_OBSTACLES.get(stage, []):
        if obs['type'] == 'cylinder':
            c = mpatches.Circle(
                (obs['x'], obs['y']), obs['radius'],
                linewidth=1, edgecolor='#444444', facecolor='#aaaaaa', zorder=1,
            )
            ax.add_patch(c)

        elif obs['type'] == 'box':
            cx, cy   = obs['cx'], obs['cy']
            hx, hy   = obs['half_x'], obs['half_y']
            theta_deg = math.degrees(obs.get('theta', 0.0))
            # Build rectangle in local frame (centre at origin), then rotate + translate.
            rect = mpatches.Rectangle(
                (-hx, -hy), 2 * hx, 2 * hy,
                linewidth=1, edgecolor='#444444', facecolor='#aaaaaa', zorder=1,
            )
            transform = Affine2D().rotate_deg(theta_deg).translate(cx, cy) + ax.transData
            rect.set_transform(transform)
            ax.add_patch(rect)


def save_episode_plot(
    stage: int,
    start: tuple[float, float],
    goal: tuple[float, float],
    astar_waypoints: list[tuple[float, float]],
    robot_traj: list[tuple[float, float]],
    episode: int,
    outcome: str,
    run_name: str,
    efficiency: float | str = '',
    astar_center_waypoints: list[tuple[float, float]] | None = None,
    efficiency_center: float | str = '',
    plots_dir: str = './path_plots',
    timestamp: str = '',
) -> None:
    """Save top-down path comparison plot to {plots_dir}/{run_name}/."""
    out_dir = os.path.join(plots_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f'ep{episode:05d}_{outcome}.png')

    fig, ax = plt.subplots(figsize=(9, 7))

    _draw_stage(ax, stage)

    # Goal acceptance circle (drawn below path lines)
    goal_circle = mpatches.Circle(
        goal, REACH_THRESHOLD,
        linewidth=1.5, linestyle='--', edgecolor='#d62728', facecolor='none',
        zorder=2, label=f'Goal region (r={REACH_THRESHOLD} m)',
    )
    ax.add_patch(goal_circle)

    # Filled circle matching the Gazebo cylinder marker (radius = REACH_THRESHOLD)
    goal_fill = mpatches.Circle(
        goal, REACH_THRESHOLD,
        linewidth=0, facecolor='#f9a3a3', alpha=0.45,
        zorder=2, label='_nolegend_',
    )
    ax.add_patch(goal_fill)

    # A* reference path — region metric (blue)
    astar_endpoint: tuple[float, float] | None = None
    if len(astar_waypoints) >= 2:
        reg_len = _path_len(astar_waypoints)
        xs, ys = zip(*astar_waypoints)
        ax.plot(xs, ys, color='#1f77b4', linewidth=2.0, zorder=3,
                label=f'A* region — {reg_len:.2f} m')
        astar_endpoint = astar_waypoints[-1]
    elif len(astar_waypoints) == 1:
        ax.plot(astar_waypoints[0][0], astar_waypoints[0][1],
                'o', color='#1f77b4', markersize=6, zorder=3,
                label='A* region (start at goal)')
        astar_endpoint = astar_waypoints[0]

    # A* centre path — centre metric (purple, dashed)
    if astar_center_waypoints and len(astar_center_waypoints) >= 2:
        cen_len = _path_len(astar_center_waypoints)
        cxs, cys = zip(*astar_center_waypoints)
        ax.plot(cxs, cys, color='#9467bd', linewidth=1.5, linestyle='--',
                zorder=3, label=f'A* centre — {cen_len:.2f} m')
        cex, cey = astar_center_waypoints[-1]
        ax.plot(cex, cey, 'D', color='#9467bd', markersize=7, zorder=7,
                label='_nolegend_')

    # A* region endpoint marker (validity visible on plot; distance in title)
    ep_dist_str = ''
    if astar_endpoint is not None:
        ex, ey = astar_endpoint
        ep_dist = math.hypot(ex - goal[0], ey - goal[1])
        valid = ep_dist <= REACH_THRESHOLD
        valid_tag = 'valid' if valid else 'OUTSIDE'
        ep_color  = '#1f77b4' if valid else '#e31a1c'
        ax.plot(ex, ey, 's', color=ep_color, markersize=8, zorder=7,
                label='_nolegend_')
        ep_dist_str = f'  |  ep_d={ep_dist:.3f} m [{valid_tag}]'

    # Actual robot trajectory
    if len(robot_traj) >= 2:
        traj_len = _path_len(list(robot_traj))
        rx, ry = zip(*robot_traj)
        ax.plot(rx, ry, color='#ff7f0e', linewidth=2.0, zorder=4,
                label=f'Trajectory — {traj_len:.2f} m')

    # Start and goal markers (drawn on top)
    ax.plot(start[0], start[1], 'o', color='#2ca02c', markersize=10,
            zorder=6, label='Start')
    ax.plot(goal[0], goal[1], '*', color='#d62728', markersize=14,
            zorder=6, label='Goal center')

    # Title
    ts_str = f'\n{timestamp}' if timestamp else ''
    ax.set_title(
        f'Episode {episode}  —  {outcome}\n'
        f'Stage {stage}  |  {run_name}{ts_str}',
        fontsize=10,
    )
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    arena = STAGE_ARENAS.get(stage, {})
    if arena:
        pad = 0.3
        ax.set_xlim(arena['x_min'] - pad, arena['x_max'] + pad)
        ax.set_ylim(arena['y_min'] - pad, arena['y_max'] + pad)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
