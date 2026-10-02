"""
Neural Models Lab: XOR (2-2-1), symmetry, activations, and a 3-class extension.
Run:  python3 neural_lab.py          (CPU only, a few seconds)
"""
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)      # float64 makes finite-difference checks clean

# ---------------------------------------------------------------- data
X  = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y  = torch.tensor([[0.], [1.], [1.], [0.]])           # XOR targets (binary)
Y3 = torch.tensor([0, 1, 1, 2])                       # 3-class targets (Task 5)

ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
LR   = {"sigmoid": 1.0, "tanh": 0.5, "relu": 0.1}     # learning rates chosen by trial
STEPS = 5000
bce = nn.BCEWithLogitsLoss()                          # sigmoid + binary cross-entropy (mean over 4 examples)
ce  = nn.CrossEntropyLoss()                           # softmax + cross-entropy (mean over 4 examples)


def make_model(act="tanh", n_out=1, init="random", seed=0):
    """2 inputs -> 2 hidden units (act) -> n_out logits.  model[0]=layer 1, model[2]=output layer."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 2), ACTS[act](), nn.Linear(2, n_out))
    if init != "random":                              # symmetry experiment: identical weights
        value = 0.0 if init == "zeros" else 0.5
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(value)
    return model


def train(model, loss_fn, target, lr, steps=STEPS, snap_steps=(0, 1, 10, 100, 1000)):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    out = {"snaps": {}}
    for step in range(steps):
        opt.zero_grad()
        logits = model(X)                    # forward pass
        loss = loss_fn(logits, target)       # scalar loss
        loss.backward()                      # reverse-mode autodiff (backprop)
        if step == 0:
            out["loss0"] = loss.item()
            out["grad_norm0"] = model[0].weight.grad.norm().item()   # Euclidean norm of dL/dW1
        if step in snap_steps:
            out["snaps"][step] = model[0].weight.detach().clone()
        opt.step()                           # parameters change here
    with torch.no_grad():
        out["loss"] = loss_fn(model(X), target).item()
    return out


def binary_eval(model):
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
    labels = (probs > 0.5).double()
    return probs.squeeze(), labels.squeeze(), bool((labels == Y).all())


def success_rate(act, n_seeds=20):
    ok = 0
    for s in range(n_seeds):
        m = make_model(act, seed=s); train(m, bce, Y, LR[act]); ok += binary_eval(m)[2]
    return ok, n_seeds


# ================================================================ Task 4, Part A
print("=" * 70, "\nPART A: basic learning check (tanh hidden units)\n", "=" * 70, sep="")
def learns_xor(act, seed):
    m = make_model(act, seed=seed)
    train(m, bce, Y, LR[act])
    return binary_eval(m)[2]

SEED = next((s for s in range(50) if learns_xor("tanh", s)), 0)
print(f"seed used: {SEED}  (first seed in 0..49 for which tanh learns XOR; 2 hidden units can get stuck)")
model = make_model("tanh", seed=SEED)
res = train(model, bce, Y, LR["tanh"])
probs, labels, ok = binary_eval(model)
print(f"initial loss = {res['loss0']:.4f}   final loss = {res['loss']:.6f}")
print("probabilities :", [round(v, 4) for v in probs.tolist()])
print("labels        :", labels.tolist(), " targets:", Y.squeeze().tolist(), " all correct:", ok)

# ================================================================ Part B
print("\n" + "=" * 70, "\nPART B: backprop check (gradient of first-layer weights)\n", "=" * 70, sep="")
model = make_model("tanh", seed=SEED)
loss = bce(model(X), Y); loss.backward()
g = model[0].weight.grad.clone()                       # = dL/dW1, shape (2,2)
print("model[0].weight.grad  (dL/dW1) =\n", g)

# (i) mean loss => gradient is the average of per-example gradients
acc = torch.zeros_like(g)
for i in range(4):
    model.zero_grad()
    bce(model(X[i:i + 1]), Y[i:i + 1]).backward()
    acc += model[0].weight.grad
print("average of 4 per-example gradients equals grad of mean loss:", torch.allclose(g, acc / 4))

# (ii) central finite differences
W = model[0].weight
fd = torch.zeros(2, 2); eps = 1e-6
for i in range(2):
    for j in range(2):
        with torch.no_grad():
            W[i, j] += eps;     lp = bce(model(X), Y).item()
            W[i, j] -= 2 * eps; lm = bce(model(X), Y).item()
            W[i, j] += eps
        fd[i, j] = (lp - lm) / (2 * eps)
print("finite-difference gradient =\n", fd)
print("max |autograd - finite difference| =", (g - fd).abs().max().item())

# ================================================================ Part C
print("\n" + "=" * 70, "\nPART C: symmetry experiment (tanh)\n", "=" * 70, sep="")
for init in ("zeros", "const"):
    label = "all weights = 0" if init == "zeros" else "all weights = 0.5 (identical, non-zero)"
    m = make_model("tanh", init=init)
    r = train(m, bce, Y, LR["tanh"], steps=2000)
    print(f"\n-- {label} --")
    for step, W1 in sorted(r["snaps"].items()):
        print(f"step {step:4d}  W1 row0 = {[round(v, 5) for v in W1[0].tolist()]}  "
              f"row1 = {[round(v, 5) for v in W1[1].tolist()]}  identical: {torch.equal(W1[0], W1[1])}")
    pr, lb, ok = binary_eval(m)
    print(f"final loss = {r['loss']:.4f}   probs = {[round(v, 3) for v in pr.tolist()]}   4/4 correct: {ok}")

# ================================================================ Part D
print("\n" + "=" * 70, "\nPART D: activation experiment (random init, same seed)\n", "=" * 70, sep="")
print(f"{'activation':10s} {'lr':>4s} {'final loss':>11s} {'4/4 correct?':>13s} {'||grad W1||2 (step 0)':>22s} {'solved over 20 seeds':>22s}")
for act in ("sigmoid", "tanh", "relu"):
    m = make_model(act, seed=SEED); r = train(m, bce, Y, LR[act]); ok = binary_eval(m)[2]
    s_ok, n = success_rate(act)
    print(f"{act:10s} {LR[act]:4.1f} {r['loss']:11.5f} {str(ok):>13s} {r['grad_norm0']:22.5f} {s_ok:>19d}/{n}")

print("\nDiagnostics at initialisation (pre-activation a, activation h, local derivative):")
for act in ("sigmoid", "relu"):
    m = make_model(act, seed=SEED)
    with torch.no_grad():
        a = m[0](X); h = m[1](a)
    deriv = h * (1 - h) if act == "sigmoid" else (a > 0).double()
    print(f"[{act}] pre-activations a:\n{a}\n      local derivative f'(a):\n{deriv}")

# ================================================================ Task 5
print("\n" + "=" * 70, "\nTASK 5: three-class extension (2-2-3, tanh, cross-entropy)\n", "=" * 70, sep="")
for s in range(5):
    m3 = make_model("tanh", n_out=3, seed=s); r3 = train(m3, ce, Y3, 0.5, steps=3000)
    with torch.no_grad(): pred = m3(X).argmax(1).tolist()
    print(f"seed {s}: final loss {r3['loss']:.5f}  predicted {pred}  target {Y3.tolist()}")

m3 = make_model("tanh", n_out=3, seed=SEED); train(m3, ce, Y3, 0.5, steps=3000)
print("\nshape of output weight matrix:", tuple(m3[2].weight.shape), "| logits per example:", m3(X).shape[1])
logits = m3(X); logits.retain_grad()
loss = ce(logits, Y3); loss.backward()
p = torch.softmax(logits.detach(), dim=1)
y = nn.functional.one_hot(Y3, 3).double()
print("class probabilities for the four inputs (rows: 00, 01, 10, 11):\n", p)
print("softmax vector for input (0,1):", p[1].tolist(), " sum =", p[1].sum().item())
print("logits.grad * N  ==  p - y :", torch.allclose(logits.grad * 4, p - y))

z = logits.detach()
print("\nShift test: add the same constant to all logits")
print("  torch.softmax(z+100) - softmax(z), max abs diff:", (torch.softmax(z + 100, 1) - torch.softmax(z, 1)).abs().max().item())
naive = lambda t: torch.exp(t) / torch.exp(t).sum(1, keepdim=True)
print("  naive softmax(z+100)  finite?", bool(torch.isfinite(naive(z + 100)).all()))
print("  naive softmax(z+1000) finite?", bool(torch.isfinite(naive(z + 1000)).all()), "(exp overflows -> nan)")
zs = z + 1000
stable = torch.exp(zs - zs.max(1, keepdim=True).values); stable = stable / stable.sum(1, keepdim=True)
print("  max-subtracted softmax(z+1000) matches softmax(z):", torch.allclose(stable, torch.softmax(z, 1)))
