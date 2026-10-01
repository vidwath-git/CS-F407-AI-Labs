# AI Laboratory: Logical Reasoning for Planning

Files: `planner.py` (planner), `tests.py` (Tests A–C). Run with `python3 planner.py` and `python3 tests.py`.

---
## Task 0: The planning problem

**Layout:** A — B — C (A and C are not directly connected).

- **(a) Initial state I** = { At(Robot,A), At(Package,A) }
- **(b) Goal G** = { At(Package,C) }
- **(c, d) Actions, with preconditions and effects:**

| Action | Preconditions | Effects (added) | Effects (deleted) |
|---|---|---|---|
| Move(A,B) | At(Robot,A) | At(Robot,B) | At(Robot,A) |
| Move(B,A) | At(Robot,B) | At(Robot,A) | At(Robot,B) |
| Move(B,C) | At(Robot,B) | At(Robot,C) | At(Robot,B) |
| Move(C,B) | At(Robot,C) | At(Robot,B) | At(Robot,C) |
| PickUp(Package,x), x ∈ {A,B,C} | At(Robot,x), At(Package,x) | Holding(Package) | At(Package,x) |
| Drop(Package,x), x ∈ {A,B,C} | At(Robot,x), Holding(Package) | At(Package,x) | Holding(Package) |

**Question: which actions are applicable in I?**
- **PickUp(Package,A) is applicable.** Its preconditions At(Robot,A) and At(Package,A) are both in I, so I ⊨ Preconditions(a).
- **Drop(Package,C) is not applicable.** It needs At(Robot,C) and Holding(Package), and neither is true in I.

Think About It: an action is not applicable just because it is in the list. It is applicable only if all its preconditions are satisfied in the current state; this is where logical reasoning enters planning.

## Task 1: Plan by hand

| State | Facts | Produced by |
|---|---|---|
| S0 | At(Robot,A), At(Package,A) | |
| S1 | At(Robot,A), Holding(Package) | PickUp(Package,A) |
| S2 | At(Robot,B), Holding(Package) | Move(A,B) |
| S3 | At(Robot,C), Holding(Package) | Move(B,C) |
| S4 | At(Robot,C), At(Package,C) | Drop(Package,C) |

S4 ⊨ G, so this is a valid plan.

Note: the lab's example sequence (Move(A,B), PickUp(Package,B), …) is not valid, because the package is at A, not B, so PickUp(Package,B) has an unsatisfied precondition. The package must be picked up at A.

## Task 2: Where the ideas appear in the program

| Idea | Where in `planner.py` |
|---|---|
| Preconditions: when is an action applicable? | `Action.is_applicable`: `pos_pre <= state and neg_pre.isdisjoint(state)` |
| Effects: how does the state change? | `Action.apply`: `(state - neg_eff) \| pos_eff` |
| Goal: when does planning terminate? | `goal <= new_state` in `plan()` |
| BFS: how are alternatives explored? | a FIFO `deque` as the frontier; `came_from` records visited states and the path |
| No plan | the frontier becomes empty, so `plan()` returns `None` and the program prints "No plan found" |

## Task 3: Test results

| Test | Initial state | Goal | Plan found? | Plan | Valid? |
|---|---|---|---|---|---|
| **A**: solvable | At(Robot,A), At(Package,A) | At(Package,C) | Yes | PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | **Yes**: replaying every action ends in {At(Package,C), At(Robot,C)} |
| **B**: PickUp removed | same | At(Package,C) | **No**: "No plan found" | none | n/a: the planner does not invent an action |
| **C**: extra irrelevant action Wander(A,B) | same | At(Package,C) | Yes | same 4-step plan | **Yes** |
| **C (contrast)**: goal At(Robot,C), no PickUp | same | At(Robot,C) | Yes | Move(A,B), Move(B,C) | Yes, but the final state still has At(Package,A), so At(Package,C) is **not** achieved |

Test C shows the planner does not treat "robot at C" as "package at C". The extra move leaves At(Package,A) true, and only At(Package,C) satisfies the goal.

## Task 4: Logic and search

```
Current state
   ↓
Check action preconditions        (is S ⊨ Pre(a)?)
   ↓
Keep only the applicable actions
   ↓
Generate successor state          S' = (S − neg effects) ∪ pos effects
   ↓
Search over alternatives          (BFS: queue unseen successors)
   ↓
Goal?                             (G ⊆ S')  yes → return plan;  no → expand next state
```

**How they work together:** logic determines what is possible; search determines what to try. The logical part tests S ⊨ Preconditions(a) and computes the effect of each action, so only legal transitions are generated. The search part decides which of those legal transitions to explore and in what order (breadth-first), and stops when a state satisfies the goal. Without the logic, search would explore impossible moves; without search, logic would give legal moves but no way to find a sequence that reaches the goal.

## Reflection questions

2. **Error if preconditions were not checked:** the planner would accept Drop(Package,C) in the initial state and "deliver" a package the robot never picked up. (I tested this by disabling the check: the planner returned the one-step plan `[Drop(Package,C)]`, which fails validation.)
3. **Why a plan that looks reasonable may be invalid:** validity depends on every action's preconditions holding in the exact state where it is executed, not on whether the sequence sounds sensible. For example, Move(A,B) then PickUp(Package,B) looks plausible but fails because the package is at A.
6. **Where logical reasoning is used:** in the applicability test S ⊨ Preconditions(a), in computing successor states from effects, and in the goal test S ⊨ G.
7. **Relation to search:** planning is search over a state space. States are sets of propositions, actions are the edges (usable only when applicable), the initial state is the start node, and any state satisfying G is a goal node. BFS, as studied in the previous module, finds a shortest plan.
