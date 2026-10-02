"""NumPy re-implementation of the 2-2-1 XOR network with hand-written backprop.
Used to cross-check the maths and the qualitative PyTorch results."""
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
Y = np.array([[0],[1],[1],[0]], float)

ACT = {
 "sigmoid": (lambda a: 1/(1+np.exp(-a)), lambda h, a: h*(1-h)),
 "tanh":    (np.tanh,                    lambda h, a: 1-h**2),
 "relu":    (lambda a: np.maximum(a,0),  lambda h, a: (a>0).astype(float)),
}

def init(seed, zero=False):
    r = np.random.default_rng(seed)
    if zero:
        return dict(W1=np.zeros((2,2)), b1=np.zeros(2), W2=np.zeros((1,2)), b2=np.zeros(1))
    return dict(W1=r.normal(0,1,(2,2)), b1=r.normal(0,1,2), W2=r.normal(0,1,(1,2)), b2=r.normal(0,1,1))

def forward(P, act):
    f, _ = ACT[act]
    A1 = X @ P["W1"].T + P["b1"]; H = f(A1)
    z = (H @ P["W2"].T + P["b2"])           # logits, shape (4,1)
    p = 1/(1+np.exp(-z))
    loss = -np.mean(Y*np.log(p+1e-12) + (1-Y)*np.log(1-p+1e-12))
    return A1, H, z, p, loss

def backward(P, act):
    _, df = ACT[act]
    A1, H, z, p, loss = forward(P, act)
    dz = (p - Y)/len(X)                      # dL/dz for mean BCE-with-logits
    g = {"W2": dz.T @ H, "b2": dz.sum(0)}
    dH = dz @ P["W2"]
    dA1 = dH * df(H, A1)
    g["W1"] = dA1.T @ X; g["b1"] = dA1.sum(0)
    return g, loss

def train(P, act, lr, steps, record_early=1):
    early = None; hist = []
    for t in range(steps):
        g, loss = backward(P, act)
        if t == record_early: early = np.linalg.norm(g["W1"])
        if t < 4: hist.append(P["W1"].copy())
        for k in P: P[k] -= lr*g[k]
    return loss, early, hist

def acc(P, act):
    p = forward(P, act)[3]
    return int(((p>0.5)==Y).all()), p.ravel()

if __name__ == "__main__":
    # gradient check (central finite differences)
    P = init(0); g, _ = backward(P, "tanh"); eps = 1e-6; worst = 0
    for k in P:
        for idx in np.ndindex(P[k].shape):
            old = P[k][idx]
            P[k][idx] = old+eps; lp = forward(P, "tanh")[4]
            P[k][idx] = old-eps; lm = forward(P, "tanh")[4]
            P[k][idx] = old
            worst = max(worst, abs((lp-lm)/(2*eps) - g[k][idx]))
    print("max |analytic - finite diff| =", worst)

    # activation experiment, several seeds
    print("\nact      lr   seeds solved/20   median final loss   median early ||grad W1||")
    for act, lr in [("sigmoid",1.0),("tanh",0.5),("relu",0.1)]:
        res = []
        for s in range(20):
            P = init(s); l0 = forward(P, act)[4]
            loss, early, _ = train(P, act, lr, 5000)
            res.append((acc(P, act)[0], loss, early))
        r = np.array(res)
        print(f"{act:8s} {lr:4} {int(r[:,0].sum()):3d}/20             {np.median(r[:,1]):.4f}              {np.median(r[:,2]):.4f}")

    # one example run, seed 0, each activation
    print("\nseed 0 detail")
    for act, lr in [("sigmoid",1.0),("tanh",0.5),("relu",0.1)]:
        P = init(0); l0 = forward(P, act)[4]
        loss, early, _ = train(P, act, lr, 5000); ok, p = acc(P, act)
        print(f"{act:8s} init loss {l0:.4f} final {loss:.5f} 4/4={bool(ok)} probs={np.round(p,3)} early grad {early:.4f}")

    # symmetry
    P = init(0, zero=True); _, _, h = train(P, "tanh", 0.5, 2000)
    print("\nzero init W1 after steps 0..3 :", [w.tolist() for w in h])
    print("final W1 rows identical:", np.allclose(P["W1"][0], P["W1"][1]), "| final loss", round(forward(P,"tanh")[4],4),
          "| probs", np.round(forward(P,"tanh")[3].ravel(),3))
    # zero-init: W1 never changes?  W2 gradient is zero initially because H = tanh(0)=0, and dH=0 since W2=0

# ------------------------------------------------------------------
# Extra checks (appended): Adam reliability, constant-init symmetry, 3-class softmax
# ------------------------------------------------------------------
def train_adam(P, act, lr, steps, record_early=1):
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, e = 0.9, 0.999, 1e-8; early = None
    for t in range(1, steps+1):
        g, loss = backward(P, act)
        if t-1 == record_early: early = np.linalg.norm(g["W1"])
        for k in P:
            m[k] = b1*m[k]+(1-b1)*g[k]; v2[k] = b2*v2[k]+(1-b2)*g[k]**2
            P[k] -= lr*(m[k]/(1-b1**t))/(np.sqrt(v2[k]/(1-b2**t))+e)
    return loss, early

if __name__ == "__main__":
    print("\n--- Adam, 3000 steps, 50 seeds ---")
    for lr in (0.01, 0.05, 0.1):
        row = []
        for act in ("sigmoid","tanh","relu"):
            ok = 0
            for s in range(50):
                P = init(s); train_adam(P, act, lr, 3000); ok += acc(P, act)[0]
            row.append(f"{act} {ok}/50")
        print(f"lr={lr}:", "  ".join(row))

    print("\n--- Symmetry with identical NONZERO constant init (all weights 0.5), tanh, SGD lr 0.5 ---")
    P = dict(W1=np.full((2,2),.5), b1=np.full(2,.5), W2=np.full((1,2),.5), b2=np.full(1,.5))
    for step in range(3001):
        if step in (0,1,10,100,3000):
            print(f"step {step:4d} W1 rows:", np.round(P["W1"],5).tolist(), "identical:", np.allclose(P["W1"][0],P["W1"][1]), "loss %.4f" % forward(P,"tanh")[4])
        g,_ = backward(P,"tanh")
        for k in P: P[k] -= 0.5*g[k]
    print("final probs", np.round(forward(P,"tanh")[3].ravel(),3))

    print("\n--- 3-class extension (2-2-3, tanh, softmax CE, Adam lr 0.05) ---")
    Y3 = np.array([0,1,1,2]); Y3oh = np.eye(3)[Y3]
    def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)
    for seed in (0,1,2):
        r = np.random.default_rng(seed)
        Q = dict(W1=r.normal(0,1,(2,2)), b1=r.normal(0,1,2), W2=r.normal(0,1,(3,2)), b2=r.normal(0,1,3))
        m = {k: np.zeros_like(v) for k,v in Q.items()}; vv = {k: np.zeros_like(v) for k,v in Q.items()}
        for t in range(1, 2001):
            H = np.tanh(X@Q["W1"].T+Q["b1"]); p = softmax(H@Q["W2"].T+Q["b2"])
            loss = -np.mean(np.log(p[np.arange(4), Y3]))
            dz = (p-Y3oh)/4                                  # (p - y) / N
            g = {"W2": dz.T@H, "b2": dz.sum(0)}; dA = (dz@Q["W2"])*(1-H**2)
            g["W1"] = dA.T@X; g["b1"] = dA.sum(0)
            for k in Q:
                m[k]=.9*m[k]+.1*g[k]; vv[k]=.999*vv[k]+.001*g[k]**2
                Q[k] -= .05*(m[k]/(1-.9**t))/(np.sqrt(vv[k]/(1-.999**t))+1e-8)
        print(f"seed {seed}: loss {loss:.5f} predicted classes {p.argmax(1).tolist()} (target {Y3.tolist()}) W2 shape {Q['W2'].shape}")
    print("probabilities (seed 2):\n", np.round(p,3), "\nrow sums:", p.sum(1))
    z = H@Q["W2"].T+Q["b2"]
    naive = np.exp(z+100); print("naive exp(z+100) -> overflow? ", np.isfinite(naive).all(), "| stable softmax(z+100) - softmax(z) max diff:", np.abs(softmax(z+100)-softmax(z)).max())
    # check gradient p-y numerically for one example
    print("p - y for input (0,1):", np.round(p[1]-Y3oh[1],4))
