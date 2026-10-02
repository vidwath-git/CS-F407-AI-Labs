# AI Laboratory: Goal-Based Agent for Warehouse Navigation

File: `warehouse_agent.py`. Run with `python3 warehouse_agent.py`.

---
## Task 1: Understanding the problem

1. **Environment:** a 7 × 21 grid warehouse. `#` cells are shelving units (obstacles), `.` cells are free, `S` is the start, `G` is the goal. The environment is fully observable (the whole map is known), deterministic (a move always has its intended effect), static (the layout does not change while the vehicle moves) and discrete.
2. **Goal:** move the vehicle from the loading bay S to the dispatch area G along a collision-free path (ideally with the fewest moves).
3. **Actions:** Up, Down, Left, Right, each moving one grid square (cost 1). A move is invalid if it would enter an obstacle or leave the map.
4. **Information the agent must maintain:** its current position (row, col) (the state); the map (which cells are free); the goal position; and, during planning, which positions have already been explored and how each was reached (to avoid cycles and to recover the path).
5. **Why goal-based and not simple reflex:** a simple reflex agent maps the current percept directly to an action by fixed condition–action rules (e.g. "if blocked on the right, turn down"), with no objective and no look-ahead, so it can get trapped in dead ends or loops. This agent has an explicit goal (reach G), uses a model of how actions change its position, and chooses actions by considering the consequences of sequences of actions (search) to find a route that reaches the goal.

**Think About It: a warehouse twice as large.** Breadth-first search would still be *correct* (complete and optimal) but more expensive. If each dimension doubles, the number of cells grows 4× (here 66 free cells would become roughly 260), and BFS may explore a large fraction of them, using memory proportional to the explored region. Difficulties that could arise: time and memory for much larger maps; a need for informed search (A* with a Manhattan-distance heuristic) to focus the search towards the goal; replanning if the map changes or other vehicles/people move (a dynamic environment); partially known maps; different move costs; and several vehicles needing coordination.

## Task 2: Design of the agent

| Component | Design |
|---|---|
| Environment | the grid map; `Warehouse.is_free()` tells which positions are allowed |
| Current state | the vehicle's position (row, col), initially S |
| Goal | the position of G |
| Actions | Up, Down, Left, Right |
| Transition model | `Warehouse.result(pos, action)`: the new position, or invalid if blocked |
| Decision-making component | `GoalBasedAgent.plan()`: breadth-first search from the current state to the goal; the agent then executes the resulting actions |

**Block diagram**

```
                    +-------------------------------------------+
                    |                 AGENT                     |
                    |                                           |
   percept          |   +-----------------+    +-------------+  |
 (position, map) -->|-->|  current state  |--->|  GOAL: G    |  |
                    |   +--------+--------+    +------+------+  |
                    |            |                    |         |
                    |            v                    v         |
                    |   +------------------------------------+  |
                    |   |  Decision making: BFS search       |  |
                    |   |  (uses model: result(state, action))|  |
                    |   +----------------+-------------------+  |
                    |                    | plan = sequence of   |
                    |                    v actions              |
                    +--------------------+----------------------+
                                         | action (Up/Down/Left/Right)
                                         v
                    +-------------------------------------------+
                    |  ENVIRONMENT: warehouse grid              |
                    |  position changes; obstacles block moves  |
                    +-------------------------------------------+
                                         |
                                         +--> new position (back to the percept)
```

## Task 3: The program and its results

**Algorithm and why it is appropriate:** the program uses **breadth-first search (BFS)**. All moves cost 1, so BFS is *complete* (it finds a path whenever one exists, and reports "No path exists" when the frontier empties) and *optimal* (the first time it reaches G it has used the fewest moves). The warehouse is small, so its time and memory cost are not a concern. A set of already-visited positions ensures each cell is expanded at most once, which prevents infinite loops.

**Output on the lab map**

```
Path found: 20 moves
Actions: Right Right Right Down Right Right Right Up Right Right Right Right Right Right Right Right Right Right Right Right

#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```
(`*` marks the path.)

**Validation**
- *Path validity:* replaying the 20 actions with the environment's transition function never hits an obstacle and ends exactly on G.
- *Optimality:* the Manhattan distance from S (1,1) to G (1,19) is 18, but the wall at column 6 in row 1 forces a detour through row 2 (down and up), which adds 2 moves, so 20 is the minimum. BFS guarantees this.
- *Failure case:* on a map where G is walled off (`#S#G#`), the program prints "No path exists from S to G." and stops.
