"""Simple planning agent (Logical Planning lab): BFS over sets of propositions."""
from collections import deque


class Action:
    def __init__(self, name, pos_pre=None, neg_pre=None, pos_eff=None, neg_eff=None):
        self.name = name
        self.pos_pre = frozenset(pos_pre or [])
        self.neg_pre = frozenset(neg_pre or [])
        self.pos_eff = frozenset(pos_eff or [])
        self.neg_eff = frozenset(neg_eff or [])

    def is_applicable(self, state):
        """S |= Preconditions(a)"""
        return self.pos_pre <= state and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        """S' = (S - neg_effects) | pos_effects"""
        return (state - self.neg_eff) | self.pos_eff

    def __repr__(self):
        return self.name


def move(a, b):
    return Action(f"Move({a},{b})", [f"At(Robot,{a})"], None, [f"At(Robot,{b})"], [f"At(Robot,{a})"])

def pickup(loc):
    return Action(f"PickUp(Package,{loc})", [f"At(Robot,{loc})", f"At(Package,{loc})"], None,
                  ["Holding(Package)"], [f"At(Package,{loc})"])

def drop(loc):
    return Action(f"Drop(Package,{loc})", [f"At(Robot,{loc})", "Holding(Package)"], None,
                  [f"At(Package,{loc})"], ["Holding(Package)"])


INITIAL_STATE = frozenset({"At(Robot,A)", "At(Package,A)"})
GOAL = frozenset({"At(Package,C)"})

MOVES = [move("A", "B"), move("B", "A"), move("B", "C"), move("C", "B")]
PICKUPS = [pickup(l) for l in "ABC"]
DROPS = [drop(l) for l in "ABC"]
ACTIONS = MOVES + PICKUPS + DROPS          # PickUp actions are ENABLED


def plan(initial_state, actions, goal):
    """BFS. Returns (plan_actions, states, expanded); plan_actions is None if no plan."""
    if goal <= initial_state:
        return [], [initial_state], 0
    frontier = deque([initial_state])
    came_from = {initial_state: None}
    expanded = 0
    while frontier:
        current = frontier.popleft()
        expanded += 1
        for action in actions:
            if not action.is_applicable(current):
                continue
            new_state = action.apply(current)
            if new_state not in came_from:
                came_from[new_state] = (current, action)
                if goal <= new_state:
                    acts, states = reconstruct(came_from, new_state)
                    return acts, states, expanded
                frontier.append(new_state)
    return None, None, expanded


def reconstruct(came_from, goal_state):
    actions, states, state = [], [], goal_state
    while came_from[state] is not None:
        prev, action = came_from[state]
        actions.append(action); states.append(state); state = prev
    states.append(state); actions.reverse(); states.reverse()
    return actions, states


def validate(initial_state, plan_actions, goal):
    """Independent checker: replays the plan, re-checking every precondition."""
    state = initial_state
    for i, a in enumerate(plan_actions, 1):
        if not (a.pos_pre <= state and a.neg_pre.isdisjoint(state)):
            return False, f"step {i} {a}: preconditions not satisfied in {sorted(state)}"
        state = (state - a.neg_eff) | a.pos_eff
    return goal <= state, f"final state {sorted(state)}"


if __name__ == "__main__":
    acts, states, n = plan(INITIAL_STATE, ACTIONS, GOAL)
    if acts is None:
        print("No plan found.")
    else:
        print(f"S0 = {sorted(states[0])}")
        for i, (a, s) in enumerate(zip(acts, states[1:]), 1):
            print(f"  {i}. {a}\n  S{i} = {sorted(s)}")
        print(f"Plan length {len(acts)}, states expanded {n}")
