"""
Task 3: Tabular Q-learning for the warehouse environment (NumPy only, no RL library, no path search).

Run:  python task3_qlearning.py
Outputs: printed summary, task3_training_curves.png, task3_greedy_trajectory.png
"""
import numpy as np

# ----------------------------------------------------------------------------
# Specification (taken directly from the lab)
# ----------------------------------------------------------------------------
ROWS, COLS = 5, 5
OBSTACLES = {(0, 2), (1, 2), (3, 2), (4, 2)}
START, GOAL = (2, 0), (2, 4)
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Wait"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]   # codes 0..4
N_ACTIONS = 5
N_STATES = ROWS * COLS

ALPHA, GAMMA = 0.2, 0.95
EPISODES = 1500
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 1200
MAX_STEPS = 40
SEED = 7


# ----------------------------------------------------------------------------
# 1. Environment
# ----------------------------------------------------------------------------
class Warehouse:
    """
    State: robot position (r, c).  Deterministic transitions.
    Reward (mutually exclusive): +10 goal, -1 invalid move (obstacle/boundary), -0.1 otherwise
    (including Wait).
    step() returns separate flags:
        terminated: the robot reached the goal (true end of the task)
        truncated : the 40-step collection cutoff was hit without reaching the goal
    """

    def __init__(self, max_steps=MAX_STEPS):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.pos = START
        self.t = 0
        return self.pos

    def set_position(self, pos):
        """Test mode helper: place the robot anywhere and restart the step counter."""
        self.pos = pos
        self.t = 0

    def step(self, action):
        self.t += 1
        dr, dc = MOVES[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        invalid = False
        if action != 4 and (not (0 <= nr < ROWS and 0 <= nc < COLS) or (nr, nc) in OBSTACLES):
            invalid = True            # stay in place
        else:
            self.pos = (nr, nc)       # for Wait, (nr, nc) == current position

        terminated = self.pos == GOAL
        if terminated:
            reward = 10.0
        elif invalid:
            reward = -1.0
        else:
            reward = -0.1
        truncated = (not terminated) and self.t >= self.max_steps
        return self.pos, reward, terminated, truncated, invalid


def state_index(pos):
    """Q-table row for cell (r, c): 5r + c."""
    return 5 * pos[0] + pos[1]


# ----------------------------------------------------------------------------
# 2. Epsilon-greedy action selection
# ----------------------------------------------------------------------------
def epsilon_greedy(Q, pos, epsilon, rng):
    """With prob. epsilon pick uniformly from all 5 actions; otherwise pick a max-Q action,
    breaking ties randomly (training only)."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    q = Q[state_index(pos)]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def greedy_action(Q, pos):
    """Evaluation: argmax with FIXED tie-breaking (lowest action code). No exploration."""
    return int(np.argmax(Q[state_index(pos)]))


# ----------------------------------------------------------------------------
# 3. One Q-learning update
# ----------------------------------------------------------------------------
def q_update(Q, pos, action, reward, next_pos, terminated, alpha=ALPHA, gamma=GAMMA):
    """
    y = r                              if terminated   (goal reached: no continuation)
    y = r + gamma * max_a' Q(s', a')   otherwise       (including TRUNCATION)
    Only Q[s, a] changes. next_pos is the state BEFORE any reset.
    """
    s, s2 = state_index(pos), state_index(next_pos)
    target = reward if terminated else reward + gamma * Q[s2].max()
    Q[s, action] += alpha * (target - Q[s, action])
    return target


# ----------------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------------
def epsilon_at(episode):
    """Linear decay 1 -> 0.05 over the first 1200 episodes, then constant 0.05."""
    frac = min(episode / EPS_DECAY_EPISODES, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def train(episodes=EPISODES, seed=SEED):
    rng = np.random.default_rng(seed)
    env = Warehouse()
    Q = np.zeros((N_STATES, N_ACTIONS))
    log = {k: [] for k in ("return", "success", "timeout", "invalid", "epsilon", "steps")}

    for ep in range(episodes):
        eps = epsilon_at(ep)
        pos = env.reset()
        ep_return, ep_invalid = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            a = epsilon_greedy(Q, pos, eps, rng)
            next_pos, r, terminated, truncated, invalid = env.step(a)
            q_update(Q, pos, a, r, next_pos, terminated)   # uses next_pos from BEFORE reset
            ep_return += r
            ep_invalid += invalid
            pos = next_pos

        log["return"].append(ep_return)
        log["success"].append(float(terminated))
        log["timeout"].append(float(truncated))
        log["invalid"].append(ep_invalid)
        log["epsilon"].append(eps)
        log["steps"].append(env.t)
    return Q, {k: np.array(v) for k, v in log.items()}


# ----------------------------------------------------------------------------
# 5. Greedy evaluation (epsilon = 0, no updates)
# ----------------------------------------------------------------------------
def evaluate_greedy(Q, start=START):
    env = Warehouse()
    env.set_position(start)
    path, invalid_total = [start], 0
    terminated = truncated = False
    while not (terminated or truncated):
        a = greedy_action(Q, env.pos)
        pos, _, terminated, truncated, invalid = env.step(a)
        invalid_total += invalid
        path.append(pos)
    return {"success": terminated, "timeout": truncated, "path": path,
            "steps": len(path) - 1, "invalid": invalid_total}


# ----------------------------------------------------------------------------
# 6. Test mode: set a position, choose one action, inspect the outcome (no training)
# ----------------------------------------------------------------------------
def test_step(pos, action):
    env = Warehouse()
    env.set_position(pos)
    next_pos, r, term, trunc, invalid = env.step(action)
    return {"from": pos, "action": ACTION_NAMES[action], "to": next_pos,
            "reward": r, "terminated": term, "truncated": trunc, "invalid": invalid}


def run_tests():
    cases = [((0, 0), 0, (0, 0), -1.0, False),
             ((1, 1), 3, (1, 1), -1.0, False),
             ((2, 1), 3, (2, 2), -0.1, False),
             ((2, 1), 4, (2, 1), -0.1, False),
             ((2, 3), 3, (2, 4), 10.0, True)]
    for pos, a, exp_pos, exp_r, exp_term in cases:
        out = test_step(pos, a)
        assert out["to"] == exp_pos and abs(out["reward"] - exp_r) < 1e-9 \
            and out["terminated"] == exp_term, out
    # 40-step cutoff reports truncation, not termination
    env = Warehouse(); env.set_position((2, 1))
    for _ in range(MAX_STEPS):
        _, _, term, trunc, _ = env.step(4)
    assert trunc and not term
    # Synthetic TD update: Q=0.4, r=1, max Q(s')=2, alpha=0.2, gamma=0.95
    Q = np.zeros((N_STATES, N_ACTIONS))
    s, s2 = (2, 1), (2, 2)
    Q[state_index(s), 3] = 0.4
    Q[state_index(s2), 0] = 2.0
    q_update(Q, s, 3, 1.0, s2, terminated=False)
    assert abs(Q[state_index(s), 3] - 0.9) < 1e-9           # 0.4 + 0.2*(2.9-0.4)
    Q[state_index(s), 3] = 0.4
    q_update(Q, s, 3, 5.0, s2, terminated=True)
    assert abs(Q[state_index(s), 3] - 1.32) < 1e-9          # 0.4 + 0.2*(5-0.4), ignores Q(s')
    Q[state_index(s), 3] = 0.4
    q_update(Q, s, 3, 1.0, s2, terminated=False)            # truncation also bootstraps
    assert abs(Q[state_index(s), 3] - 0.9) < 1e-9
    print("All environment and update tests passed.")


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(log, window=50, path="task3_training_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(moving_average(log["return"], window))
    ax[0].set(title=f"Training return (moving avg, window={window})", xlabel="Episode", ylabel="Return")
    ax[1].plot(moving_average(log["success"], window))
    ax[1].set(title=f"Goal reached (moving avg, window={window})", xlabel="Episode", ylabel="Fraction", ylim=(0, 1.05))
    ax[2].plot(log["epsilon"])
    ax[2].set(title="Epsilon", xlabel="Episode", ylabel="epsilon")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_trajectory(path_cells, out="task3_greedy_trajectory.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in range(ROWS):
        for c in range(COLS):
            color = "#555555" if (r, c) in OBSTACLES else "white"
            ax.add_patch(plt.Rectangle((c, ROWS - 1 - r), 1, 1, facecolor=color, edgecolor="black"))
    for name, (r, c) in (("S", START), ("G", GOAL)):
        ax.text(c + 0.5, ROWS - 1 - r + 0.5, name, ha="center", va="center", fontsize=16, weight="bold")
    xs = [c + 0.5 for r, c in path_cells]
    ys = [ROWS - 1 - r + 0.5 for r, c in path_cells]
    ax.plot(xs, ys, "o-", color="tab:blue", lw=2)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y + 0.25, str(i), ha="center", fontsize=9, color="tab:blue")
    ax.set(xlim=(0, COLS), ylim=(0, ROWS), aspect="equal", title="Learned greedy trajectory (numbers = time step)")
    ax.set_xticks(np.arange(COLS) + 0.5, range(COLS))
    ax.set_yticks(np.arange(ROWS) + 0.5, range(ROWS - 1, -1, -1))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_tests()

    Q, log = train()
    w = 50
    print(f"\nTrained {EPISODES} episodes (seed {SEED}, alpha={ALPHA}, gamma={GAMMA}).")
    print(f"Last {w} training episodes: success rate {log['success'][-w:].mean():.2f}, "
          f"mean return {log['return'][-w:].mean():.2f}, "
          f"timeouts {int(log['timeout'][-w:].sum())}, "
          f"mean invalid moves {log['invalid'][-w:].mean():.2f}")
    print(f"First {w} training episodes: success rate {log['success'][:w].mean():.2f}, "
          f"mean return {log['return'][:w].mean():.2f}")

    res = evaluate_greedy(Q)
    print(f"\nGreedy evaluation from {START}: success={res['success']}, timeout={res['timeout']}, "
          f"steps={res['steps']}, invalid moves={res['invalid']}")
    print("Trajectory:", " -> ".join(map(str, res["path"])))

    plot_training(log, window=w)
    plot_trajectory(res["path"])
    print("\nSaved task3_training_curves.png and task3_greedy_trajectory.png")