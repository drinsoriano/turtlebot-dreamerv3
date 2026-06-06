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
