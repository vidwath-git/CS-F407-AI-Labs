# AI Laboratory: Search and A*

Files: `search_lab.py` (A*, BFS, the four maps), `tests.py` (extra validation). Run with `python3 search_lab.py` and `python3 tests.py`.

---
## Task 0: The search problem

| Component | Specification |
|---|---|
| State S | A grid position (row, col) of a free cell (not `#`) |
| Actions A | Up (−1,0), Down (+1,0), Left (0,−1), Right (0,+1) |
| Transition T | T((r,c), a) = (r+dr, c+dc) if that cell is inside the map and is not `#`; otherwise the action is invalid |
| Initial state s0 | the cell marked `S`: (1,1) |
| Goal G | { the cell marked `G` } = {(7,15)} |
| Cost c | 1 per move, so the cost of a path = its number of steps |

- **(a) Information needed to specify a state:** only the robot's row and column. The warehouse layout is fixed, so it is part of the problem, not the state.
- **(b) An invalid action:** one that would move into an obstacle `#` or off the map.
- **(c) Deterministic?** Yes. Each action in a state leads to exactly one next state, with no uncertainty.
- **(d) A solution:** a sequence of valid moves from (1,1) to (7,15). An optimal solution is one with minimum total cost (fewest moves).

## Task 1: Design of the agent
1. **State in Python:** a tuple `(row, col)`.
2. **Warehouse:** the ASCII map parsed into a list of strings; `parse_grid` also finds the positions of `S` and `G`.
3. **Valid actions:** `neighbors(grid, state)` tries the four moves and keeps those where `is_free` is true (in bounds and not `#`).
4. **Goal recognition:** the state popped from the frontier equals the goal position.
5. **Frontier contents:** for A*, a priority queue (min-heap) of entries `(f, h, counter, state, g)`; for BFS, a FIFO queue of states. Also stored: `came_from` (parent of each state) and `best_g` (cheapest known cost to each state).
6. **Path reconstruction:** follow `came_from` from the goal back to the start, then reverse.
7. **Reported at termination:** whether a solution was found, the path, the path length, and the number of states expanded.

## Task 3: Tests

| Test | Map | Result |
|---|---|---|
| 1. Original warehouse | the lab map | Found. Path length **40**, **64** states expanded (A* and BFS). Valid path (checked by a separate function that verifies start, goal, and that every step is one move onto a free cell) |
| 2. Trivial | `#SG##` | Found. Path length **1**, 2 states expanded |
| 3. No solution | G enclosed by walls | "No solution found." after 9 states expanded (the whole reachable region), with no infinite loop |
| 4. Alternative paths | two routes from S to G | Found. Length **10**, which is a shortest path: S=(1,1) and G=(5,7) are 4+6 = 10 apart, so no path can be shorter (Manhattan distance is a lower bound). A* expands 11 states, BFS 25 |

Additional checks (`tests.py`): on 2000 random 10×10 grids with 25% obstacles (1418 solvable), A* with Manhattan returned the same path length as BFS in all 1418 cases, and both reported failure on the same unsolvable grids.

## Task 4: Where the concepts appear in the code

| Concept | Where |
|---|---|
| State | the tuple `(row, col)`, e.g. `current`, `start`, `goal` |
| Action | the four `(dr, dc)` pairs in `moves` inside `neighbors` |
| Transition | `(r + dr, c + dc)` in `neighbors`, filtered by `is_free` |
| Goal test | `if current == goal:` in `astar` (and in `bfs`) |
| g(n) | `g` stored in each frontier entry; `new_g = g + 1`; best known cost in `best_g` |
| h(n) | `heuristic(nxt, goal)`, a function such as `manhattan` |
| f(n) | `new_g + h` when pushing onto the heap |
| Frontier | `frontier`, a list managed by `heapq` (priority queue) |
| Visited states | `best_g` and `came_from` dictionaries |
| Path reconstruction | `reconstruct_path(came_from, current)` |

- **(a) Frontier data structure:** a binary min-heap (`heapq`) of tuples.
- **(b) Selecting the next state:** `heapq.heappop` returns the entry with the smallest `f` (ties broken by smaller `h`, then insertion order).
- **(c) Where h is calculated:** when a successor is generated, `heuristic(nxt, goal)` (and once for the start).
- **(d) Is f = g + h calculated explicitly?** Yes, `new_g + h` is computed when each entry is pushed; it is the heap's priority.
- **(e) Preventing repeated exploration:** a successor is only pushed if it is new or reached with a lower g (`new_g < best_g[nxt]`), and entries whose g is worse than `best_g` are skipped when popped (`if g > best_g[current]: continue`).

**Design note on tie-breaking:** the first version broke ties on f only by insertion order. In an empty 10×10 room all cells on the way have the same f, so A* with Manhattan expanded 90 of 100 cells, no better than BFS. Adding `h` to the heap key (prefer states closer to the goal) reduced this to **18**, with the same optimal length 17. This is the version in `search_lab.py`.

## Task 5: A* vs BFS (original warehouse)

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

- **(a)** Both found a solution.
- **(b)** Yes, both found paths of length 40; both are optimal.
- **(c)** Neither: both expanded all 64 free cells, so on this map A* has no advantage.
- **(d) Why A* *may* expand fewer:** it orders states by f = g + h, so it prefers states that appear to lead towards the goal, whereas BFS explores all states in order of distance from the start. In this warehouse the corridors wind away from the goal: Manhattan distance from S to G is 20, but the true cost is 40. The heuristic is therefore very optimistic and does not prune anything, and A* has to explore everything as BFS does. On maps where the straight-line direction is useful A* does expand fewer (same lengths):

| Map | BFS expanded | A* expanded |
|---|---|---|
| Alternative-paths map | 25 | 11 |
| Empty 10×10 room | 90 | 18 |

A* is therefore not better by default; its benefit depends on how informative the heuristic is.

## Task 6: Heuristic investigation

**Original warehouse**

| Heuristic | Solution found | Path length | States expanded |
|---|---|---|---|
| h = 0 | Yes | 40 | 64 |
| Manhattan | Yes | 40 | 64 |
| Euclidean | Yes | 40 | 64 |
| 2 × Manhattan | Yes | 40 | 68 |

On the given warehouse all four give the same result, because the maze is essentially one winding corridor with dead ends, so there is little choice to be influenced. To see the differences I also tested an empty 10×10 room (optimal length 17), and 1418 random solvable 10×10 grids:

| Heuristic | Empty room: length / expanded | Random grids: optimal in |
|---|---|---|
| h = 0 | 17 / 90 | 1418 / 1418 (always) |
| Manhattan | 17 / 18 | 1418 / 1418 |
| Euclidean | 17 / 74 | 1418 / 1418 |
| 2 × Manhattan | 17 / 18 | 1049 / 1418 (74%) |
| 5 × Manhattan | not run | 821 / 1418 (58%) |

Example where 2 × Manhattan fails (S top-left, G bottom-right):
```
S..#.
.#...
...#.
..#..
###.G
```
BFS (optimal) finds 8 steps; A* with 2 × Manhattan returns **10 steps**.

**Findings (from the experiments):**
1. **h = 0:** A* still finds optimal paths but has no guidance, so it behaves like uniform-cost/breadth-first search and expands as many states as BFS.
2. **Euclidean:** still admissible, so still optimal, but never larger than Manhattan on a 4-connected grid, so it is less informed and expands more (74 vs 18 in the room).
3. **2 × Manhattan:** expands fewer states in the room but can overestimate the true cost (not admissible, h > h*), so A* may commit to a path too early and return a longer-than-optimal path (as in the example).

**Think About It:** when the heuristic is admissible (h ≤ h*), A* is guaranteed to return an optimal path, and the closer h is to h* the fewer states it expands. When the heuristic is too optimistic in the other direction (overestimates) A* becomes more aggressive and faster, but loses its optimality guarantee. A heuristic of 0 is safe but uninformative.

## Final reflection (questions 1–3)

**1. Why formulate the problem before writing the algorithm?** The algorithm is generic: it only needs to know the states, the actions, the transitions, the goal test, and the costs. If these are not defined, the code cannot be written correctly or tested, and there is nothing to check results against. Formulating the problem (Task 0) also exposes decisions such as what counts as a state and what makes a move invalid, which the program must follow exactly.

**2. In what sense is A\* "informed"?** Blind searches like BFS use only the structure of the problem (how far a state is from the start). A* also uses domain knowledge in the heuristic h(n), an estimate of how far each state is from the goal, and chooses to expand the state with the lowest f(n) = g(n) + h(n). This lets it direct the search towards the goal instead of exploring uniformly in all directions.

**3. Why does the choice of heuristic matter?** It determines both the efficiency and the correctness of A*. An admissible heuristic (never overestimating) guarantees an optimal path, and the closer it is to the true cost, the fewer states are expanded (Manhattan expanded 18 states in the open room versus 90 for h = 0). A heuristic that overestimates, like 2 × Manhattan, may find a path faster but can return a suboptimal one. A poor heuristic can also be useless: in the warehouse the winding corridors make Manhattan distance a weak estimate, and A* expanded every state, like BFS.
