from planner import *

def run(title, init, goal, actions):
    print(f"\n=== {title} ===")
    print("initial:", sorted(init), "| goal:", sorted(goal))
    acts, states, n = plan(init, actions, goal)
    if acts is None:
        print("result : No plan found  (states expanded:", n, ")")
        return
    print("plan   :", acts)
    print("valid  :", validate(init, acts, goal))

# Test A: original problem
run("Test A: solvable", INITIAL_STATE, GOAL, ACTIONS)

# Test B: remove PickUp
run("Test B: no PickUp action", INITIAL_STATE, GOAL, MOVES + DROPS)

# Test C: extra irrelevant action, goal is the ROBOT at C (package stays at A) vs package at C
extra = Action("Wander(A,B)", ["At(Robot,A)"], None, ["At(Robot,B)"], ["At(Robot,A)"])
run("Test C1: extra irrelevant action, goal At(Package,C)", INITIAL_STATE, GOAL, ACTIONS + [extra])
run("Test C2: no PickUp, extra moves; goal At(Package,C)", INITIAL_STATE, GOAL, MOVES + DROPS + [extra])
run("Test C3: contrast, goal At(Robot,C) is reachable w/o PickUp", INITIAL_STATE, frozenset({"At(Robot,C)"}), MOVES + [extra])

# Extra: an action is NOT applicable unless preconditions hold (Task 0)
print("\n=== Task 0 ===")
print("PickUp(Package,A) applicable in I:", pickup("A").is_applicable(INITIAL_STATE))
print("Drop(Package,C)   applicable in I:", drop("C").is_applicable(INITIAL_STATE))
# Bug-injection: ignore preconditions
class Sloppy(Action):
    def is_applicable(self, s): return True
sloppy = [Sloppy(a.name, a.pos_pre, a.neg_pre, a.pos_eff, a.neg_eff) for a in ACTIONS]
acts, _, _ = plan(INITIAL_STATE, sloppy, GOAL)
print("planner with precondition check removed finds:", acts, "->", validate(INITIAL_STATE, acts, GOAL))
