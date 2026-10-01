"""Extra validation for search_lab.py"""
import random
from search_lab import *

def valid_path(grid, path, start, goal):
    """Independent checker: starts at S, ends at G, each step is one free-cell move."""
    if path[0] != start or path[-1] != goal:
        return False
    return all(is_free(grid, *b) and manhattan(a, b) == 1 for a, b in zip(path, path[1:]))

# 1. Path validity on all maps
for name, m in [("warehouse", WAREHOUSE_MAP), ("trivial", TRIVIAL_MAP), ("alt", ALTERNATIVE_PATHS_MAP)]:
    g, s, goal = parse_grid(m)
    for alg, f in [("A*", lambda: astar(g, s, goal)), ("BFS", lambda: bfs(g, s, goal))]:
        p, c, e = f()
        print(f"{name:10s} {alg:4s} valid={valid_path(g, p, s, goal)} steps={len(p)-1}")

# 2. How many free cells does the warehouse have?
g, s, goal = parse_grid(WAREHOUSE_MAP)
free = sum(ch != "#" for row in g for ch in row)
print("\nwarehouse free cells:", free, "| start", s, "| goal", goal)
print("Manhattan(S,G) =", manhattan(s, goal), "but true shortest path = 40")

# 3. Open room: where A* should beat BFS
open_map = "\n".join(["#" * 12] + ["#" + "S" + "." * 9 + "#"] + ["#" + "." * 10 + "#"] * 7 + ["#" + "." * 9 + "G" + "#"] + ["#" * 12])
g, s, goal = parse_grid(open_map)
print("\nOpen 10x10 room:")
for name, f in [("BFS", lambda: bfs(g, s, goal)), ("A* zero h", lambda: astar(g, s, goal, zero_heuristic)),
                ("A* Manhattan", lambda: astar(g, s, goal, manhattan)), ("A* Euclid", lambda: astar(g, s, goal, euclidean)),
                ("A* 2xManhattan", lambda: astar(g, s, goal, manhattan_times_2))]:
    p, c, e = f(); print(f"  {name:15s} steps={len(p)-1:3d} expanded={e}")

# 4. Optimality on random grids: A* (admissible h) must equal BFS length; weighted h may not
random.seed(0)
def rand_map(n=10, p=0.25):
    cells = [["#" if random.random() < p else "." for _ in range(n)] for _ in range(n)]
    cells[0][0], cells[n-1][n-1] = "S", "G"
    return "\n".join("".join(r) for r in cells)

stats = {"solvable": 0, "astar_opt": 0, "euclid_opt": 0, "h2_opt": 0, "h5_opt": 0}
h5 = lambda a, b: 5 * manhattan(a, b)
worst = None
for _ in range(2000):
    g, s, goal = parse_grid(rand_map())
    pb, cb, _ = bfs(g, s, goal)
    if pb is None:
        assert astar(g, s, goal)[0] is None   # failure agreement
        continue
    stats["solvable"] += 1
    stats["astar_opt"] += astar(g, s, goal)[1] == cb
    stats["euclid_opt"] += astar(g, s, goal, euclidean)[1] == cb
    c2 = astar(g, s, goal, manhattan_times_2)[1]; stats["h2_opt"] += c2 == cb
    c5 = astar(g, s, goal, h5)[1]; stats["h5_opt"] += c5 == cb
    if worst is None and c5 > cb: worst = (rand_map.__name__, c5, cb)
print("\nRandom 10x10 grids:", stats)

# 5. Smallest random map where 2 x Manhattan returns a LONGER path than BFS
random.seed(1)
best = None
for _ in range(20000):
    n = random.choice([5, 6, 7])
    g, s, goal = parse_grid(rand_map(n, 0.3))
    pb, cb, _ = bfs(g, s, goal)
    if pb is None: continue
    c2 = astar(g, s, goal, manhattan_times_2)[1]
    if c2 > cb and (best is None or n < best[0]):
        best = (n, rand_map.__name__, g, s, goal, cb, c2)
        if n == 5: break
n, _, g, s, goal, cb, c2 = best
print("\nCounter-example for inadmissible h = 2 x Manhattan:")
print("\n".join(g)); print(f"BFS (optimal) = {cb} steps, A* with 2xManhattan = {c2} steps")
