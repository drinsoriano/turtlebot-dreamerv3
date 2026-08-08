"""
stage_map.py — Static occupancy grids and A* path planner for each training stage.

Obstacle geometry is hard-coded from the Gazebo SDF files under
turtlebot3_gazebo/models/turtlebot3_dqn_world/.  No runtime SDF parsing is needed.

Usage (called from Env in turtle.py):
    from envs.stage_map import get_grid, astar_path_length
    grid, arena = get_grid(stage)
    planned_metres = astar_path_length(grid, arena, start_xy, goal_xy)
"""
from __future__ import annotations

import heapq
import math

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

RESOLUTION  = 0.05   # metres per grid cell
ROBOT_RADIUS = 0.15  # TurtleBot3 Burger footprint ≈ 0.105 m + 0.045 m safety margin

# ── Arena bounds (from outer-wall SDF files) ───────────────────────────────────
# Stages 1-4, 7: 5 m × 5 m arena centred at origin.
# Stage 5, 8:    7.5 m × 7.5 m.
# Stage 6:       14 m × 7.5 m.

STAGE_ARENAS: dict[int, dict] = {
    1: {'x_min': -2.5, 'x_max': 2.5, 'y_min': -2.5, 'y_max': 2.5},
    2: {'x_min': -2.5, 'x_max': 2.5, 'y_min': -2.5, 'y_max': 2.5},
    3: {'x_min': -2.5, 'x_max': 2.5, 'y_min': -2.5, 'y_max': 2.5},
    4: {'x_min': -2.5, 'x_max': 2.5, 'y_min': -2.5, 'y_max': 2.5},
    5: {'x_min': -3.75, 'x_max': 3.75, 'y_min': -3.75, 'y_max': 3.75},
    6: {'x_min': -7.0,  'x_max': 7.0,  'y_min': -3.75, 'y_max': 3.75},
    7: {'x_min': -2.5,  'x_max': 2.5,  'y_min': -2.5,  'y_max': 2.5},
    8: {'x_min': -3.75, 'x_max': 3.75, 'y_min': -3.75, 'y_max': 3.75},
}

# ── Obstacle definitions ────────────────────────────────────────────────────────
# Each entry is a dict with key 'type':
#   cylinder: {'type':'cylinder', 'x':cx, 'y':cy, 'radius':r}
#   box:      {'type':'box', 'cx':cx, 'cy':cy, 'half_x':hx, 'half_y':hy, 'theta':t}
#
# For axis-aligned walls:
#   horizontal (long axis along world-X): half_x = half_length, half_y = half_thickness
#   vertical   (long axis along world-Y): half_x = half_thickness, half_y = half_length
#   (theta=0 with swapped half-extents is equivalent to theta=pi/2 with natural half-extents)
#
# For rotated walls (stage 3): theta is the yaw from the SDF <pose> element.
#   The check rotates the query point by -theta into the box's local frame.
#
# Obstacle inflation by ROBOT_RADIUS is applied inside build_grid().

_H = 0.075   # half-thickness of a standard 0.15 m wall

STAGE_OBSTACLES: dict[int, list[dict]] = {

    # ── Stage 1: empty arena ────────────────────────────────────────────────────
    1: [],

    # ── Stage 2: four cylinders at (±1, ±1), r=0.15 m ──────────────────────────
    # Source: obstacles_stage2/model.sdf
    2: [
        {'type': 'cylinder', 'x': -1.0, 'y': -1.0, 'radius': 0.15},
        {'type': 'cylinder', 'x': -1.0, 'y':  1.0, 'radius': 0.15},
        {'type': 'cylinder', 'x':  1.0, 'y': -1.0, 'radius': 0.15},
        {'type': 'cylinder', 'x':  1.0, 'y':  1.0, 'radius': 0.15},
    ],

    # ── Stage 3: five rotated boxes ─────────────────────────────────────────────
    # Source: obstacles_stage3/model.sdf
    # All geometries live inside a single link 'obstacle'; positions come from
    # <collision><pose> directly (model pose is identity).
    3: [
        # single_wall:  pos (0, 1.7), size 1.2×0.3, yaw=0.8 rad
        {'type': 'box', 'cx':  0.0, 'cy':  1.70, 'half_x': 0.60, 'half_y': 0.15, 'theta':  0.80},
        # little_square: pos (-0.2, -1.45), size 0.2×0.2, yaw=0
        {'type': 'box', 'cx': -0.2, 'cy': -1.45, 'half_x': 0.10, 'half_y': 0.10, 'theta':  0.00},
        # compo_wall_1: pos (-0.2, -0.2), size 1.2×0.3, yaw=0.8 rad
        {'type': 'box', 'cx': -0.2, 'cy': -0.20, 'half_x': 0.60, 'half_y': 0.15, 'theta':  0.80},
        # compo_wall_2: pos (-0.8, 0), size 1.2×0.3, yaw=-0.75 rad
        {'type': 'box', 'cx': -0.8, 'cy':  0.00, 'half_x': 0.60, 'half_y': 0.15, 'theta': -0.75},
        # compo_wall_3: pos (0.6, -0.33), size 1.2×0.3, yaw=-0.75 rad
        {'type': 'box', 'cx':  0.6, 'cy': -0.33, 'half_x': 0.60, 'half_y': 0.15, 'theta': -0.75},
    ],

    # ── Stage 4: seven axis-aligned 1.0 m × 0.15 m walls ───────────────────────
    # Source: obstacles_stage4/model.sdf
    4: [
        # inner_wall_1: pos (-2.0, -1.5), horizontal
        {'type': 'box', 'cx': -2.00, 'cy': -1.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_2: pos (-0.5, -2.0), vertical (yaw -pi/2)
        {'type': 'box', 'cx': -0.50, 'cy': -2.000, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_3: pos (1.0, -1.0), vertical (yaw pi/2)
        {'type': 'box', 'cx':  1.00, 'cy': -1.000, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_4: pos (1.2, 1.9), vertical (yaw -pi/2)
        {'type': 'box', 'cx':  1.20, 'cy':  1.900, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_5: pos (1.9, 0.4), horizontal
        {'type': 'box', 'cx':  1.90, 'cy':  0.400, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_6: pos (-0.5, 1.5), horizontal
        {'type': 'box', 'cx': -0.50, 'cy':  1.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_7: pos (-1.2, 0.092), vertical (yaw -pi/2)
        {'type': 'box', 'cx': -1.20, 'cy':  0.092, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
    ],

    # ── Stage 5: ten 1.0 m × 0.15 m walls (obstacles_stage5, no offset) ────────
    # Source: obstacles_stage5/model.sdf, spawned at (0,0) in stage 5 world
    5: [
        # inner_wall_1: pos (-2.0, -1.5), horizontal
        {'type': 'box', 'cx': -2.00, 'cy': -1.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_1_alternative: pos (2.0, -1.5), horizontal
        {'type': 'box', 'cx':  2.00, 'cy': -1.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_1_alternative_2: pos (-2.5, 2.0), horizontal
        {'type': 'box', 'cx': -2.50, 'cy':  2.000, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_1_alternative_3: pos (2.5, 2.0), horizontal
        {'type': 'box', 'cx':  2.50, 'cy':  2.000, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_2: pos (-0.5, -2.0), vertical (yaw -pi/2)
        {'type': 'box', 'cx': -0.50, 'cy': -2.000, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_3: pos (1.0, -1.0), vertical (yaw pi/2)
        {'type': 'box', 'cx':  1.00, 'cy': -1.000, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_4: pos (1.2, 1.9), vertical (yaw -pi/2)
        {'type': 'box', 'cx':  1.20, 'cy':  1.900, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_5: pos (1.9, 0.4), horizontal
        {'type': 'box', 'cx':  1.90, 'cy':  0.400, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_6: pos (-0.5, 1.5), horizontal
        {'type': 'box', 'cx': -0.50, 'cy':  1.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_7: pos (-1.2, 0.092), vertical (yaw -pi/2)
        {'type': 'box', 'cx': -1.20, 'cy':  0.092, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
    ],

    # ── Stage 6: two obstacle groups, each spawned at a model-frame offset ──────
    # Source: obstacles_stage5 spawned at (3.2, -0.5)  +  obstacles_stage6 at (-3.0, -0.5)
    # Each world pos = local_pos + model_offset.
    6: [
        # ── Group A: obstacles_stage5 at offset (3.2, -0.5) ────────────────────
        {'type': 'box', 'cx':  1.20, 'cy': -2.000, 'half_x': 0.50, 'half_y': _H,   'theta': 0.0},  # w1
        {'type': 'box', 'cx':  5.20, 'cy': -2.000, 'half_x': 0.50, 'half_y': _H,   'theta': 0.0},  # w1_alt
        {'type': 'box', 'cx':  0.70, 'cy':  1.500, 'half_x': 0.50, 'half_y': _H,   'theta': 0.0},  # w1_alt2
        {'type': 'box', 'cx':  5.70, 'cy':  1.500, 'half_x': 0.50, 'half_y': _H,   'theta': 0.0},  # w1_alt3
        {'type': 'box', 'cx':  2.70, 'cy': -2.500, 'half_x': _H,   'half_y': 0.50, 'theta': 0.0},  # w2
        {'type': 'box', 'cx':  4.20, 'cy': -1.500, 'half_x': _H,   'half_y': 0.50, 'theta': 0.0},  # w3
        {'type': 'box', 'cx':  4.40, 'cy':  1.400, 'half_x': _H,   'half_y': 0.50, 'theta': 0.0},  # w4
        {'type': 'box', 'cx':  5.10, 'cy': -0.100, 'half_x': 0.50, 'half_y': _H,   'theta': 0.0},  # w5
        {'type': 'box', 'cx':  2.70, 'cy':  1.000, 'half_x': 0.50, 'half_y': _H,   'theta': 0.0},  # w6
        {'type': 'box', 'cx':  2.00, 'cy': -0.408, 'half_x': _H,   'half_y': 0.50, 'theta': 0.0},  # w7
        # ── Group B: obstacles_stage6 at offset (-3.0, -0.5) ───────────────────
        # inner_wall_1: (1.5, 1.5) + (-3.0, -0.5) = (-1.5, 1.0), size 1.0×0.15
        {'type': 'box', 'cx': -1.50, 'cy':  1.000, 'half_x': 0.50,  'half_y': _H,    'theta': 0.0},
        # inner_wall_small: (1.35, -1.0) + offset = (-1.65, -1.5), size 0.5×0.15
        {'type': 'box', 'cx': -1.65, 'cy': -1.500, 'half_x': 0.25,  'half_y': _H,    'theta': 0.0},
        # inner_wall_2: (1.0, -1.5) + offset = (-2.0, -2.0), size 1.5×0.15, vertical
        {'type': 'box', 'cx': -2.00, 'cy': -2.000, 'half_x': _H,    'half_y': 0.75,  'theta': 0.0},
        # inner_wall_horizontal_small: (0.30, 2.0) + offset = (-2.7, 1.5), size 0.5×0.15, vertical
        {'type': 'box', 'cx': -2.70, 'cy':  1.500, 'half_x': _H,    'half_y': 0.25,  'theta': 0.0},
        # inner_wall_3: (-1.5, 0.5) + offset = (-4.5, 0.0), size 2.0×0.15, horizontal
        {'type': 'box', 'cx': -4.50, 'cy':  0.000, 'half_x': 1.00,  'half_y': _H,    'theta': 0.0},
        # inner_wall_3_alternative: (3.0, -2.0) + offset = (0.0, -2.5), size 1.5×0.15, horizontal
        {'type': 'box', 'cx':  0.00, 'cy': -2.500, 'half_x': 0.75,  'half_y': _H,    'theta': 0.0},
        # inner_wall_3_alternative_2: (-3.0, -2.0) + offset = (-6.0, -2.5), size 1.5×0.15, horizontal
        {'type': 'box', 'cx': -6.00, 'cy': -2.500, 'half_x': 0.75,  'half_y': _H,    'theta': 0.0},
        # inner_wall_big2: (-1.4, -1.0) + offset = (-4.4, -1.5), size 1.0×0.15, horizontal
        {'type': 'box', 'cx': -4.40, 'cy': -1.500, 'half_x': 0.50,  'half_y': _H,    'theta': 0.0},
        # inner_wall_4: (-0.75, 1.5) + offset = (-3.75, 1.0), size 1.0×0.15, horizontal
        {'type': 'box', 'cx': -3.75, 'cy':  1.000, 'half_x': 0.50,  'half_y': _H,    'theta': 0.0},
        # inner_wall_5: (-0.3, -1.7) + offset = (-3.3, -2.2), size 1.2×0.15, vertical
        {'type': 'box', 'cx': -3.30, 'cy': -2.200, 'half_x': _H,    'half_y': 0.60,  'theta': 0.0},
        # inner_wall_5_alternative: (2.0, 3.0) + offset = (-1.0, 2.5), size 1.2×0.15, vertical
        {'type': 'box', 'cx': -1.00, 'cy':  2.500, 'half_x': _H,    'half_y': 0.60,  'theta': 0.0},
        # inner_wall_5_alternative_2: (-2.0, 3.0) + offset = (-5.0, 2.5), size 1.2×0.15, vertical
        {'type': 'box', 'cx': -5.00, 'cy':  2.500, 'half_x': _H,    'half_y': 0.60,  'theta': 0.0},
        # inner_wall_6: (1.0, 0.2) + offset = (-2.0, -0.3), size 0.8×0.15, vertical
        {'type': 'box', 'cx': -2.00, 'cy': -0.300, 'half_x': _H,    'half_y': 0.40,  'theta': 0.0},
    ],

    # ── Stage 7: six 1.0 m × 0.15 m walls ──────────────────────────────────────
    # Source: obstacles_stage7/model.sdf
    7: [
        # inner_wall_1: pos (-1.0, -1.5), horizontal
        {'type': 'box', 'cx': -1.00, 'cy': -1.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_2: pos (0.5, -1.0), vertical (yaw -pi/2)
        {'type': 'box', 'cx':  0.50, 'cy': -1.000, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_3: pos (1.5, -0.5), vertical (yaw pi/2)
        {'type': 'box', 'cx':  1.50, 'cy': -0.500, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_4: pos (-1.5, 1.0), vertical (yaw -pi/2)
        {'type': 'box', 'cx': -1.50, 'cy':  1.000, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
        # inner_wall_6: pos (-0.5, 0.5), horizontal
        {'type': 'box', 'cx': -0.50, 'cy':  0.500, 'half_x': 0.50, 'half_y': _H, 'theta': 0.0},
        # inner_wall_7: pos (0, 1.5), vertical (yaw -pi/2)
        {'type': 'box', 'cx':  0.00, 'cy':  1.500, 'half_x': _H, 'half_y': 0.50, 'theta': 0.0},
    ],

    # ── Stage 8: empty expanded arena (obstacles_stage8/model.sdf is empty) ─────
    8: [],
}


# ── Grid builder ───────────────────────────────────────────────────────────────

def build_grid(
    stage: int,
    resolution: float = RESOLUTION,
    robot_radius: float = ROBOT_RADIUS,
) -> tuple[np.ndarray, dict]:
    """Build a boolean occupancy grid for *stage*.

    Returns (grid, arena) where grid[row, col] == True means occupied.
    Cells outside the arena bounds are implicitly occupied (A* never steps there).
    """
    arena = STAGE_ARENAS[stage]
    w = int(round((arena['x_max'] - arena['x_min']) / resolution)) + 1
    h = int(round((arena['y_max'] - arena['y_min']) / resolution)) + 1
    grid = np.zeros((h, w), dtype=bool)

    # Block cells within robot_radius of the outer arena boundary so the
    # planner never routes the robot's centre too close to the outer walls.
    margin = int(math.ceil(robot_radius / resolution))
    grid[:margin, :]  = True
    grid[-margin:, :] = True
    grid[:, :margin]  = True
    grid[:, -margin:] = True

    for obs in STAGE_OBSTACLES.get(stage, []):
        if obs['type'] == 'cylinder':
            r2 = (obs['radius'] + robot_radius) ** 2
            cx, cy = obs['x'], obs['y']
            for iy in range(h):
                for ix in range(w):
                    px = arena['x_min'] + ix * resolution - cx
                    py = arena['y_min'] + iy * resolution - cy
                    if px * px + py * py <= r2:
                        grid[iy, ix] = True

        elif obs['type'] == 'box':
            hx = obs['half_x'] + robot_radius
            hy = obs['half_y'] + robot_radius
            cx, cy = obs['cx'], obs['cy']
            theta = obs.get('theta', 0.0)
            if theta == 0.0:
                # Fast path: axis-aligned box
                for iy in range(h):
                    for ix in range(w):
                        dx = abs(arena['x_min'] + ix * resolution - cx)
                        dy = abs(arena['y_min'] + iy * resolution - cy)
                        if dx <= hx and dy <= hy:
                            grid[iy, ix] = True
            else:
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                for iy in range(h):
                    for ix in range(w):
                        dx = arena['x_min'] + ix * resolution - cx
                        dy = arena['y_min'] + iy * resolution - cy
                        lx =  dx * cos_t + dy * sin_t
                        ly = -dx * sin_t + dy * cos_t
                        if abs(lx) <= hx and abs(ly) <= hy:
                            grid[iy, ix] = True

    return grid, arena


# ── A* planner ─────────────────────────────────────────────────────────────────

def astar_plan(
    grid: np.ndarray,
    arena: dict,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    reach_threshold: float = 0.4,
    resolution: float = RESOLUTION,
) -> tuple:
    """Return (path_length_metres, waypoints) for the shortest path to the goal region.

    waypoints is a list of (x, y) tuples from start to the goal-region entry cell,
    suitable for plotting.  Returns (None, []) if no path exists.

    Goal region = all free cells whose centre is within *reach_threshold* of *goal_xy*,
    matching the robot's actual stopping condition.
    Heuristic: max(0, dist_to_goal_centre − reach_threshold) — admissible.
    """
    h_cells, w_cells = grid.shape

    def to_cell(x: float, y: float) -> tuple[int, int]:
        ix = int(round((x - arena['x_min']) / resolution))
        iy = int(round((y - arena['y_min']) / resolution))
        return (iy, ix)

    def to_xy(iy: int, ix: int) -> tuple[float, float]:
        return arena['x_min'] + ix * resolution, arena['y_min'] + iy * resolution

    def free(iy: int, ix: int) -> bool:
        return 0 <= iy < h_cells and 0 <= ix < w_cells and not grid[iy, ix]

    start = to_cell(*start_xy)
    gx, gy = goal_xy
    gc = to_cell(gx, gy)
    r_cells = int(math.ceil(reach_threshold / resolution))

    # Build goal set: all free cells within reach_threshold of goal centre
    goal_set: set[tuple[int, int]] = set()
    for diy in range(-r_cells, r_cells + 1):
        for dix in range(-r_cells, r_cells + 1):
            nb = (gc[0] + diy, gc[1] + dix)
            if free(*nb):
                cx, cy = to_xy(*nb)
                if (cx - gx) ** 2 + (cy - gy) ** 2 <= reach_threshold ** 2:
                    goal_set.add(nb)

    if not goal_set:
        return None, []  # goal entirely blocked

    if start in goal_set:
        return 0.0, [start_xy]

    SQRT2 = resolution * math.sqrt(2)
    MOVES = (
        (-1,  0, resolution), (1,  0, resolution),
        ( 0, -1, resolution), (0,  1, resolution),
        (-1, -1, SQRT2),      (-1, 1, SQRT2),
        ( 1, -1, SQRT2),      ( 1, 1, SQRT2),
    )

    def heuristic(iy: int, ix: int) -> float:
        cx, cy = to_xy(iy, ix)
        return max(0.0, math.hypot(cx - gx, cy - gy) - reach_threshold)

    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple | None] = {start: None}
    heap: list = [(heuristic(*start), 0.0, start)]
    visited: set[tuple[int, int]] = set()

    while heap:
        _, g, cell = heapq.heappop(heap)
        if cell in visited:
            continue
        visited.add(cell)
        if cell in goal_set:
            # Reconstruct waypoint list by walking came_from back to start
            path: list[tuple[float, float]] = []
            cur: tuple | None = cell
            while cur is not None:
                path.append(to_xy(*cur))
                cur = came_from[cur]
            path.reverse()
            return g, path
        iy, ix = cell
        for diy, dix, cost in MOVES:
            nb = (iy + diy, ix + dix)
            if nb in visited or not free(*nb):
                continue
            # Prevent diagonal corner-cutting: both orthogonal neighbours must be free.
            if diy != 0 and dix != 0 and (not free(iy + diy, ix) or not free(iy, ix + dix)):
                continue
            ng = g + cost
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                came_from[nb] = cell
                heapq.heappush(heap, (ng + heuristic(*nb), ng, nb))

    return None, []  # no path found


def astar_path_length(
    grid: np.ndarray,
    arena: dict,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    reach_threshold: float = 0.4,
    resolution: float = RESOLUTION,
) -> float | None:
    """Return shortest path length in metres (None if no path). See astar_plan()."""
    length, _ = astar_plan(grid, arena, start_xy, goal_xy, reach_threshold, resolution)
    return length


# ── Module-level grid cache ────────────────────────────────────────────────────

_GRID_CACHE: dict[int, tuple[np.ndarray, dict]] = {}


def get_grid(stage: int) -> tuple[np.ndarray, dict]:
    """Return cached (grid, arena) for *stage*, building it on first call."""
    if stage not in _GRID_CACHE:
        _GRID_CACHE[stage] = build_grid(stage)
    return _GRID_CACHE[stage]


# ── Nonholonomic reference: Hybrid-A* (curvature-bounded, obstacle-aware) ──────
# A reference path that respects the robot's minimum turning radius AND routes
# around obstacles — a *fair* efficiency denominator on every stage, unlike the
# holonomic grid A* above. Forward-only (curvature-bounded) arc motion primitives +
# an analytic arc shot to the goal + a holonomic-with-obstacles heuristic, memoised
# over the start/goal pairs that repeat across episodes (fixed goals, discrete spawn
# points). Post-hoc metric only — never fed into training.


def is_occupied(grid: np.ndarray, arena: dict, x: float, y: float,
                resolution: float = RESOLUTION) -> bool:
    """True if world point (x, y) falls on an occupied (already ROBOT_RADIUS-inflated)
    cell or outside the arena. Uses the same round-to-cell convention as astar_plan,
    so a point-check on the inflated grid is equivalent to a footprint-check."""
    h_cells, w_cells = grid.shape
    ix = int(round((x - arena['x_min']) / resolution))
    iy = int(round((y - arena['y_min']) / resolution))
    if not (0 <= iy < h_cells and 0 <= ix < w_cells):
        return True
    return bool(grid[iy, ix])


def _norm_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _poly_len(pts: list[tuple[float, float]]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def _arc_shot(start_xyt: tuple[float, float, float],
              goal_xy: tuple[float, float],
              radius: float,
              resolution: float = RESOLUTION) -> tuple:
    """Forward-only **curve-then-straight (C·S)** arc from a directed start pose to a
    goal *point* with **free final heading** — the Hybrid-A* analytic expansion.

    Optimal bounded-curvature path to a point when the goal lies outside both
    radius-`r` turning circles tangent to the start pose. Returns (length_metres,
    waypoints) sampled at ~`resolution` (waypoints[0] == start), or (None, []) if the
    goal is inside both circles (degenerate — sub-radius range only).
    """
    x0, y0, th0 = start_xyt
    gx, gy = goal_xy
    best = None  # (total_len, center, s, phi_start, phi_td, arc, straight_len)

    for s in (+1.0, -1.0):  # +1 = left / CCW circle, -1 = right / CW circle
        cx = x0 - s * radius * math.sin(th0)
        cy = y0 + s * radius * math.cos(th0)
        d = math.hypot(gx - cx, gy - cy)
        if d < radius:                       # goal inside this circle → no C·S
            continue
        alpha = math.atan2(gy - cy, gx - cx)
        gamma = math.acos(max(-1.0, min(1.0, radius / d)))
        phi_start = math.atan2(y0 - cy, x0 - cx)
        straight_len = math.sqrt(max(0.0, d * d - radius * radius))
        for phi_td in (alpha + gamma, alpha - gamma):
            tdx = cx + radius * math.cos(phi_td)
            tdy = cy + radius * math.sin(phi_td)
            heading_td = phi_td + s * (math.pi / 2.0)        # tangent heading, dir s
            dir_tdg = math.atan2(gy - tdy, gx - tdx)
            if abs(_norm_angle(heading_td - dir_tdg)) > 1e-3:  # not forward-consistent
                continue
            arc = (s * (phi_td - phi_start)) % (2 * math.pi)   # swept angle in dir s
            if arc > 2 * math.pi - 1e-9:                       # ~2π from float wrap → 0
                arc = 0.0
            total = radius * arc + straight_len
            if best is None or total < best[0]:
                best = (total, (cx, cy), s, phi_start, phi_td, arc, straight_len)

    if best is None:
        return None, []

    total, (cx, cy), s, phi_start, phi_td, arc, straight_len = best
    wp: list[tuple[float, float]] = []
    n_arc = max(1, int(math.ceil(arc * radius / resolution)))
    for i in range(n_arc + 1):
        phi = phi_start + s * arc * (i / n_arc)
        wp.append((cx + radius * math.cos(phi), cy + radius * math.sin(phi)))
    tdx = cx + radius * math.cos(phi_td)
    tdy = cy + radius * math.sin(phi_td)
    n_str = max(1, int(math.ceil(straight_len / resolution)))
    for i in range(1, n_str + 1):
        t = i / n_str
        wp.append((tdx + (gx - tdx) * t, tdy + (gy - tdy) * t))
    return total, wp


# ── Hybrid-A* search ───────────────────────────────────────────────────────────

_HEUR_CACHE: dict = {}     # (id(grid), goal, reach) -> 2-D cost-to-goal field (m)
_HYBRID_CACHE: dict = {}   # discretised (start, goal, radius, …) -> (len, wp, status)


def _holonomic_cost_field(grid, arena, goal_xy, reach_threshold, resolution):
    """Backward Dijkstra over free cells from the goal region → metres-to-goal per
    cell (np.inf if unreachable). Admissible, obstacle-aware Hybrid-A* heuristic
    (holonomic distance ≤ nonholonomic), cached per (grid, goal, reach)."""
    key = (id(grid), round(goal_xy[0], 3), round(goal_xy[1], 3),
           round(reach_threshold, 3), round(resolution, 4))
    cached = _HEUR_CACHE.get(key)
    if cached is not None:
        return cached

    h_cells, w_cells = grid.shape

    def to_cell(x, y):
        return (int(round((y - arena['y_min']) / resolution)),
                int(round((x - arena['x_min']) / resolution)))

    def to_xy(iy, ix):
        return arena['x_min'] + ix * resolution, arena['y_min'] + iy * resolution

    def free(iy, ix):
        return 0 <= iy < h_cells and 0 <= ix < w_cells and not grid[iy, ix]

    gx, gy = goal_xy
    gc = to_cell(gx, gy)
    r_cells = int(math.ceil(reach_threshold / resolution))
    cost = np.full((h_cells, w_cells), np.inf, dtype=np.float64)
    heap = []
    for diy in range(-r_cells, r_cells + 1):
        for dix in range(-r_cells, r_cells + 1):
            iy, ix = gc[0] + diy, gc[1] + dix
            if free(iy, ix):
                cx, cy = to_xy(iy, ix)
                if (cx - gx) ** 2 + (cy - gy) ** 2 <= reach_threshold ** 2:
                    cost[iy, ix] = 0.0
                    heapq.heappush(heap, (0.0, iy, ix))

    SQRT2 = resolution * math.sqrt(2)
    MOVES = ((-1, 0, resolution), (1, 0, resolution), (0, -1, resolution),
             (0, 1, resolution), (-1, -1, SQRT2), (-1, 1, SQRT2),
             (1, -1, SQRT2), (1, 1, SQRT2))
    while heap:
        c, iy, ix = heapq.heappop(heap)
        if c > cost[iy, ix]:
            continue
        for diy, dix, w in MOVES:
            niy, nix = iy + diy, ix + dix
            if not free(niy, nix):
                continue
            if diy != 0 and dix != 0 and (not free(iy + diy, ix) or not free(iy, ix + dix)):
                continue
            nc = c + w
            if nc < cost[niy, nix]:
                cost[niy, nix] = nc
                heapq.heappush(heap, (nc, niy, nix))

    _HEUR_CACHE[key] = cost
    return cost


def _arc_points(x, y, th, s, length, radius, resolution):
    """Sample a forward arc of `length` from pose (x, y, th). s=0 straight, ±1
    left/right (curvature ±1/radius). Returns (end_x, end_y, end_th, sample_pts)
    where sample_pts excludes the start point and includes the end."""
    n = max(1, int(math.ceil(length / resolution)))
    pts = []
    if s == 0:
        for i in range(1, n + 1):
            t = length * i / n
            pts.append((x + t * math.cos(th), y + t * math.sin(th)))
        return x + length * math.cos(th), y + length * math.sin(th), th, pts
    for i in range(1, n + 1):
        t = length * i / n
        tht = th + s * t / radius
        pts.append((x + s * radius * (math.sin(tht) - math.sin(th)),
                    y - s * radius * (math.cos(tht) - math.cos(th))))
    th_end = th + s * length / radius
    return (x + s * radius * (math.sin(th_end) - math.sin(th)),
            y - s * radius * (math.cos(th_end) - math.cos(th)),
            th_end, pts)


def _reconstruct(came, key):
    """Rebuild the (x, y) waypoint polyline from start to `key` via stored arc segs."""
    segs = []
    k = key
    while k is not None:
        parent, pts = came[k]
        segs.append(pts)
        k = parent
    segs.reverse()
    out: list[tuple[float, float]] = []
    for seg in segs:
        out.extend(seg)
    return out


def hybrid_astar_plan(grid: np.ndarray, arena: dict,
                      start_xyt: tuple[float, float, float],
                      goal_xy: tuple[float, float],
                      radius: float,
                      reach_threshold: float = 0.4,
                      resolution: float = RESOLUTION,
                      n_headings: int = 24,
                      prim_len: float = 0.25,
                      max_pops: int = 40000) -> tuple:
    """Curvature-bounded, obstacle-avoiding reference path to the goal *region*.

    Returns (length_metres, waypoints, status), status ∈ {'ok','no_path',
    'planner_error'}. The path respects the minimum turning `radius` (forward-only
    arcs) and avoids inflated obstacles, so it is a fair nonholonomic efficiency
    denominator on every stage. Memoised by the discretised (start, goal, radius)
    tuple, which repeats constantly under fixed/discrete goals → amortised O(1).
    """
    try:
        h_cells, w_cells = grid.shape
        x0, y0, th0 = start_xyt
        gx, gy = goal_xy
        dth = 2 * math.pi / n_headings

        def to_cell(x, y):
            return (int(round((y - arena['y_min']) / resolution)),
                    int(round((x - arena['x_min']) / resolution)))

        def thbin(t):
            return int(round(t / dth)) % n_headings

        def skey(x, y, th):
            iy, ix = to_cell(x, y)
            return (iy, ix, thbin(th))

        ckey = (id(grid), to_cell(x0, y0), thbin(th0), to_cell(gx, gy),
                round(radius, 3), round(reach_threshold, 3),
                round(resolution, 4), n_headings, round(prim_len, 3))
        memo = _HYBRID_CACHE.get(ckey)
        if memo is not None:
            return memo

        if math.hypot(x0 - gx, y0 - gy) <= reach_threshold:
            res = (0.0, [(x0, y0)], 'ok')
            _HYBRID_CACHE[ckey] = res
            return res

        cost_field = _holonomic_cost_field(grid, arena, goal_xy, reach_threshold, resolution)

        def heur(x, y):
            iy, ix = to_cell(x, y)
            if 0 <= iy < h_cells and 0 <= ix < w_cells:
                h = cost_field[iy, ix]
                return float(h) if math.isfinite(h) else None
            return None

        h0 = heur(x0, y0)
        if h0 is None:                       # start not holonomically connected
            res = (None, [], 'no_path')
            _HYBRID_CACHE[ckey] = res
            return res

        sk0 = skey(x0, y0, th0)
        open_heap = [(h0, 0.0, x0, y0, th0)]
        gbest = {sk0: 0.0}
        came = {sk0: (None, [(x0, y0)])}

        pops = 0
        while open_heap and pops < max_pops:
            f, g, x, y, th = heapq.heappop(open_heap)
            pops += 1
            cur = skey(x, y, th)
            if g > gbest.get(cur, float('inf')) + 1e-9:
                continue

            # Node already in the goal region → done.
            if math.hypot(x - gx, y - gy) <= reach_threshold:
                res = (g, _reconstruct(came, cur), 'ok')
                _HYBRID_CACHE[ckey] = res
                return res

            # Analytic arc shot to the goal (free heading); finish if collision-free.
            shot_len, shot_wp = _arc_shot((x, y, th), goal_xy, radius, resolution)
            if shot_len is not None and shot_wp:
                trimmed = [p for p in shot_wp
                           if math.hypot(p[0] - gx, p[1] - gy) > reach_threshold]
                if not any(is_occupied(grid, arena, px, py, resolution)
                           for (px, py) in trimmed):
                    path = _reconstruct(came, cur)
                    if len(trimmed) >= 2:
                        path = path + trimmed[1:]
                    res = (g + _poly_len(trimmed), path, 'ok')
                    _HYBRID_CACHE[ckey] = res
                    return res

            for s in (0, +1, -1):
                nx, ny, nth, pts = _arc_points(x, y, th, s, prim_len, radius, resolution)
                if any(is_occupied(grid, arena, px, py, resolution) for (px, py) in pts):
                    continue
                ng = g + prim_len
                nk = skey(nx, ny, nth)
                if ng < gbest.get(nk, float('inf')) - 1e-9:
                    hh = heur(nx, ny)
                    if hh is None:
                        continue
                    gbest[nk] = ng
                    came[nk] = (cur, pts)
                    heapq.heappush(open_heap, (ng + hh, ng, nx, ny, nth))

        res = (None, [], 'no_path')
        _HYBRID_CACHE[ckey] = res
        return res
    except Exception:
        return None, [], 'planner_error'
