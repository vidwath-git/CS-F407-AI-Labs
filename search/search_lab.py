"""
Warehouse Robot Navigation - Search Lab
Implements A* and BFS on an ASCII grid warehouse map.

State representation : (row, col)
Actions               : Up, Down, Left, Right (cost 1 each)
Heuristic (for A*)    : Manhattan distance to goal (configurable)
"""

import heapq
from collections import deque
import math


def parse_grid(ascii_map):
    """Turn an ASCII map (string) into a list of rows, and find S and G."""
    rows = [line for line in ascii_map.strip("\n").split("\n")]
    start = goal = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    return rows, start, goal


def in_bounds(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[r])


def is_free(grid, r, c):
    return in_bounds(grid, r, c) and grid[r][c] != "#"


def neighbors(grid, state):
    """Return valid neighboring states (Up, Down, Left, Right)."""
    r, c = state
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(r + dr, c + dc) for dr, dc in moves if is_free(grid, r + dr, c + dc)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def zero_heuristic(a, b):
    return 0

def manhattan_times_2(a, b):
    return 2 * manhattan(a, b)


def astar(grid, start, goal, heuristic=manhattan):
    frontier = []
    counter = 0  # tie-breaker so heapq never compares states directly
    # heap entries: (f, h, counter, state, g). Ties on f are broken by smaller h
    # (prefer states closer to the goal), then by insertion order.
    h0 = heuristic(start, goal)
    heapq.heappush(frontier, (h0, h0, counter, start, 0))
    came_from = {start: None}
    best_g = {start: 0}
    expanded = 0

    while frontier:
        f, _, _, current, g = heapq.heappop(frontier)
        if g > best_g[current]:      # stale entry
            continue
        expanded += 1
        if current == goal:
            return reconstruct_path(came_from, current), g, expanded
        for nxt in neighbors(grid, current):
            new_g = g + 1
            if nxt not in best_g or new_g < best_g[nxt]:
                best_g[nxt] = new_g
                came_from[nxt] = current
                counter += 1
                h = heuristic(nxt, goal)
                heapq.heappush(frontier, (new_g + h, h, counter, nxt, new_g))
    return None, None, expanded


def bfs(grid, start, goal):
    frontier = deque([start])
    came_from = {start: None}
    expanded = 0
    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, len(path) - 1, expanded
        for nxt in neighbors(grid, current):
            if nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)
    return None, None, expanded


def reconstruct_path(came_from, current):
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def report(name, path, cost, expanded):
    print(f"--- {name} ---")
    if path is None:
        print("No solution found.")
    else:
        print("Solution found: yes")
        print(f"Path length (steps): {len(path) - 1}")
        print(f"Path: {path}")
    print(f"States expanded: {expanded}")
    print()


WAREHOUSE_MAP = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""
TRIVIAL_MAP = """
#####
#SG##
#####
"""
NO_SOLUTION_MAP = """
#######
#S....#
###.###
#...#G#
#######
"""
ALTERNATIVE_PATHS_MAP = """
#########
#S......#
#.#####.#
#.......#
#.#####.#
#......G#
#########
"""

if __name__ == "__main__":
    for map_name, ascii_map in [
        ("Test 1: Original warehouse", WAREHOUSE_MAP),
        ("Test 2: Trivial case", TRIVIAL_MAP),
        ("Test 3: No solution", NO_SOLUTION_MAP),
        ("Test 4: Alternative paths", ALTERNATIVE_PATHS_MAP),
    ]:
        print(f"===== {map_name} =====")
        grid, start, goal = parse_grid(ascii_map)
        report("A* (Manhattan)", *astar(grid, start, goal, heuristic=manhattan))
        report("BFS", *bfs(grid, start, goal))

    print("===== Task 6: Heuristic investigation (original warehouse) =====")
    grid, start, goal = parse_grid(WAREHOUSE_MAP)
    for h_name, h_func in [("h(n) = 0", zero_heuristic), ("h(n) = Manhattan", manhattan),
                           ("h(n) = Euclidean", euclidean), ("h(n) = 2 x Manhattan", manhattan_times_2)]:
        report(f"A* with {h_name}", *astar(grid, start, goal, heuristic=h_func))
