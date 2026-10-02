"""
Goal-based warehouse navigation agent.

Design (goal-based agent):
  environment : 2-D grid map (ASCII).  '#' obstacle, '.' free, 'S' start, 'G' goal
  state       : the vehicle's position (row, col)
  goal        : reach the cell marked 'G'
  actions     : Up, Down, Left, Right (one cell, cost 1)
  decision    : breadth-first search (BFS) over positions, then follow the resulting path

Why BFS?  Every move has the same cost (1), so BFS is complete (finds a path whenever one
exists) and optimal (the first path it finds uses the fewest moves). It also terminates
when no path exists, because each cell is visited at most once.
"""
from collections import deque

WAREHOUSE_MAP = """
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
"""

ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


class Warehouse:
    """The environment: a grid that says which positions are free."""
    def __init__(self, ascii_map):
        self.grid = ascii_map.strip("\n").split("\n")
        self.start = self.goal = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S": self.start = (r, c)
                if ch == "G": self.goal = (r, c)

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != "#"

    def result(self, pos, action):
        """Transition: the position after `action`, or None if it would hit an obstacle/boundary."""
        dr, dc = ACTIONS[action]
        new = (pos[0] + dr, pos[1] + dc)
        return new if self.is_free(new) else None


class GoalBasedAgent:
    def __init__(self, env):
        self.env = env
        self.position = env.start          # current state
        self.goal = env.goal               # explicit goal

    def plan(self):
        """Decision-making component: BFS from the current state to the goal.
        Returns a list of actions, or None if the goal is unreachable."""
        frontier = deque([self.position])
        parent = {self.position: None}     # position -> (previous position, action)
        while frontier:
            current = frontier.popleft()
            if current == self.goal:       # goal test
                actions = []
                while parent[current] is not None:
                    current, action = parent[current]
                    actions.append(action)
                return actions[::-1]
            for action in ACTIONS:
                nxt = self.env.result(current, action)
                if nxt is not None and nxt not in parent:
                    parent[nxt] = (current, action)
                    frontier.append(nxt)
        return None

    def act(self, actions):
        """Execute the plan one action at a time; return all visited positions."""
        trail = [self.position]
        for a in actions:
            self.position = self.env.result(self.position, a)
            trail.append(self.position)
        return trail


def draw(env, trail):
    rows = [list(r) for r in env.grid]
    for (r, c) in trail[1:-1]:
        rows[r][c] = "*"
    return "\n".join("".join(r) for r in rows)


def run(ascii_map, title):
    print(f"=== {title} ===")
    env = Warehouse(ascii_map)
    agent = GoalBasedAgent(env)
    actions = agent.plan()
    if actions is None:
        print("No path exists from S to G.\n")
        return None
    trail = agent.act(actions)
    assert agent.position == env.goal
    print(f"Path found: {len(actions)} moves")
    print("Actions:", " ".join(actions))
    print("Positions:", trail)
    print(draw(env, trail), "\n")
    return env, actions


if __name__ == "__main__":
    run(WAREHOUSE_MAP, "Warehouse map from the lab")
    run("#####\n#S#G#\n#####", "Blocked map (no path)")
