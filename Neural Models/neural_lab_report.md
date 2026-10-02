# Laboratory: Neural Models (XOR, Depth, Activations, Output Layers)

Files: `neural_lab.py` (PyTorch: all experiments), `numpy_check.py` (same network with hand-written backprop, used to cross-check the maths).
Run: `python3 neural_lab.py`.

> **Note on the numbers.** The numbers below were produced with `numpy_check.py`, a NumPy implementation of the same 2–2–1 network (hand-written backprop, mean BCE, plain gradient descent). Replace them with the output of `neural_lab.py` if your report must show PyTorch output; the qualitative conclusions are the same.

---
## Task 1: Problem specification

- **Input space** X = {0,1}², **output space** Y = {0,1}.
- **Examples:** (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0.
- **Plot:** the class-1 points (0,1) and (1,0) lie on one diagonal, the class-0 points (0,0) and (1,1) on the other.
- **Why one line cannot separate them:** any half-plane containing both (0,1) and (1,0) also contains their midpoint (0.5,0.5), which is the midpoint of (0,0) and (1,1) as well. So a line that puts both class-1 points on one side must also put at least one class-0 point on that side (convexity). XOR is not linearly separable.
- **Prediction for a single affine layer + sigmoid:** it cannot fit XOR; the best it can do is predict p = 0.5 for all four inputs, with loss ln 2 ≈ 0.693 (and at most 3 of 4 correct if forced to threshold).
  *Checked:* training it gave w = (0,0), b = 0, probabilities [0.5, 0.5, 0.5, 0.5], loss 0.6931.

**Think About It:** XOR tests the scientific claim that the *representation*, not the number of parameters, determines what a model can compute: a model whose output is a linear function of the inputs cannot represent XOR however long it trains, but a hidden nonlinear layer that re-represents the inputs can.

## Task 2: Model design and validation criteria

**Design:** 2 inputs → 2 hidden units (tanh; sigmoid and ReLU tried later) → 1 output logit with sigmoid, loss = binary cross-entropy (`BCEWithLogitsLoss`), full-batch gradient descent.

1. **Why the hidden nonlinearity is necessary:** without it, the layers collapse into one affine map, W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2), which is still a linear decision boundary. The nonlinear hidden layer re-represents the inputs (e.g. one unit ≈ "x1 OR x2", another ≈ "x1 AND x2") so that the output unit can separate the classes with a line in hidden space.
2. **Why sigmoid + BCE:** the target is a yes/no answer, so the sigmoid output is a probability in (0,1) and BCE is the negative log-likelihood of a Bernoulli variable. The logit gradient is simply p − y (no vanishing derivative from the output sigmoid).
3. **Validation criteria:** (i) final loss close to 0 (e.g. < 0.01) from an initial ≈ 0.7; (ii) all four thresholded predictions equal the labels; (iii) first-layer gradients nonzero and matching a finite-difference check; (iv) behaviour over repeated runs with different seeds (what fraction succeed).

**Think About It:** the hidden units are given no targets. What each one computes is determined only by the gradient dL/dh that backpropagation carries from the output loss through W2 and the activation derivative; the hidden features are whatever helps reduce the loss.

## Task 4: Results

### Part A: basic learning check (tanh, seed 0 of the NumPy check, 5000 steps)
Initial loss 0.767, final loss **0.0013**; probabilities [0.002, 0.999, 0.999, 0.002] → labels [0,1,1,0] = targets, **4/4 correct**.

*Important observation:* with only 2 hidden units, learning does **not** succeed for every random initialisation. Over 20 seeds (SGD, 5000 steps): sigmoid 14/20, tanh 11/20, ReLU 11/20 reached 4/4. Failed runs are stuck at a poor plateau (e.g. probabilities [0.002, 0.666, 0.666, 0.667], 3/4 correct). Therefore the code reports the seed used, and changing the seed is a justified engineering change.

### Part B: backpropagation check
- `model[0].weight.grad` is the 2×2 matrix of ∂L/∂W(1): entry (i,j) says how the loss changes per unit change of the weight from input j to hidden unit i, obtained by the chain rule ∂L/∂z · W2 · f′(a) · x.
- The loss is a **mean** over 4 examples, L = (1/4)ΣLₙ, and differentiation is linear, so ∂L/∂W = (1/4)Σ ∂Lₙ/∂W. The code verifies this by summing four per-example gradients.
- **Finite-difference check:** max |analytic − central difference| = **6.8 × 10⁻¹¹**, so the backward pass is correct.

### Part C: symmetry experiment (tanh, gradient descent lr 0.5)

| Initialisation | Rows of W(1) over training | Result |
|---|---|---|
| all weights = 0 | W(1) = [[0,0],[0,0]] at steps 0, 1, 2, 3 and at the end (identical) | loss stays **0.6931**, probabilities all 0.5 |
| all weights = 0.5 (identical, non-zero) | rows identical at steps 0, 1, 10, 100 and 3000 (e.g. both [4.254, 4.254] at step 3000), though the values change | loss 0.479, probabilities [0.002, 0.666, 0.666, 0.667] (stuck, 3/4 correct) |

**Explanation:** if both hidden units start with identical weights they compute identical outputs, so they receive identical gradients (the same dL/dh scaled by the same W2 entries and the same derivative), and gradient descent updates them identically; the rows remain identical forever, and the layer behaves like one hidden unit. With all-zero weights it is even stronger: the hidden activations are tanh(0) = 0, so the gradient for W2 is 0, and since W2 = 0 the gradient reaching W(1) is 0 as well. Only the output bias moves. Random initialisation breaks the symmetry.

### Part D: activation experiment (random init, seed 0, N(0,1), 5000 steps)

| Hidden activation | lr | Final loss | 4/4 correct? | ‖∇W(1)L‖₂ at step 1 | Solved over 20 seeds | Median ‖∇W(1)L‖ over seeds |
|---|---|---|---|---|---|---|
| Sigmoid | 1.0 | 0.348 | No | 0.0145 | 14/20 | 0.017 |
| Tanh | 0.5 | 0.0013 | **Yes** | 0.1050 | 11/20 | 0.046 |
| ReLU | 0.1 | 0.693 | No | 0.0342 | 11/20 | 0.073 |

**Interpretation:** these are four data points and one seed, so no activation is "best"; the 20-seed column shows that success depends mainly on the initialisation. What the experiment does show:
- **Sigmoid** had the smallest early gradient (median 0.017): its derivative is at most 0.25, so the signal backpropagated to the first layer is small, and it needed the largest learning rate (1.0) to train at a similar speed.
- **Tanh** is zero-centred with derivative up to 1, giving larger early gradients (median 0.046).
- **ReLU** has derivative exactly 1 for active units, which gave the largest median early gradient (0.073), but a unit whose pre-activation is negative for all four inputs has zero gradient and is "dead": the seed-0 run stayed at loss 0.693 (all outputs 0.5).
- Scientific explanation: the gradient reaching W(1) is a product of Jacobian factors (here W2 and f′(a)); engineering observation: the measured norms and success rates above, specific to this tiny setup.

**Think About It (saturation vs dead ReLU):** inspect the pre-activations a and activations h. A *saturated sigmoid* has |a| large, h close to 0 or 1, and f′(a) = h(1−h) close to 0 but never exactly 0; changing the input slightly still gives a tiny nonzero gradient. A *dead ReLU* has a ≤ 0 (for all inputs), h = 0 exactly, and derivative exactly 0 (the gradient is exactly zero, not merely small). The script prints a and f′(a) for both cases.

## Task 5: Three-class extension (2 → 2 tanh → 3 logits, softmax cross-entropy)

**Predictions before running:**
1. Output weight matrix shape **3 × 2** (3 classes × 2 hidden units), plus a bias of length 3.
2. **3 logits** per example.
3. Softmax pᵢ = e^{zᵢ}/Σⱼ e^{zⱼ}: the numerators are positive and the denominator is their sum, so Σpᵢ = 1.
4. For L = −log p_c with one-hot y: ∂L/∂zᵢ = pᵢ − yᵢ (the −1 from the log derivative applies only to the true class). For the mean loss over N examples the gradient is (pᵢ − yᵢ)/N.

**Results** (NumPy check, 2000 Adam steps; the PyTorch script uses SGD): all three tested seeds learned the task, predicted classes [0,1,1,2] = targets, final loss ≈ 10⁻⁴. Probabilities for inputs 00, 01, 10, 11 are ≈ [1,0,0], [0,1,0], [0,1,0], [0,0,1]; every row sums to 1.0. For input (0,1), p − y = [0, −0.0002, 0.0001].
Note that this 3-class problem is actually linearly separable (class depends on x1 + x2 ∈ {0,1,2}), so unlike XOR it does not strictly require the hidden layer.

**Shift diagnostic:** adding 100 to all logits changes the stable softmax output by ≈ 10⁻¹⁸ (rounding only), because the constant cancels: e^{z+c}/Σe^{z+c} = e^{z}/Σe^{z}. A naive implementation overflows for large constants (e^{1000} = inf in double precision, giving nan). Stable implementations subtract max(z) first, so the largest exponent is e⁰ = 1 and nothing overflows, with the mathematically identical result.

**Think About It:** what stays the same for tens of thousands of classes is the mathematics of the output layer: logits → softmax → cross-entropy, the p − y logit gradient, and the max-subtraction trick. What changes dramatically is the surrounding architecture: the model producing the logits (a deep network, e.g. a Transformer, over a long context, with embeddings and a very large output matrix of vocabulary × hidden size), the cost of the softmax and its gradient, and the training scale.

## Reflection questions

1. **Depth vs nonlinearity:** adding affine layers without nonlinear activations adds depth but collapses to a single affine map, so it still cannot solve XOR. The nonlinearity (hidden activation) is what makes extra units useful: a 2–2–1 network with tanh learns XOR, while linear layers cannot.
2. **Evidence that backprop supplied a *useful* signal:** the loss decreased from ≈ 0.77 to 0.0013, all four predictions became correct, and the analytic gradient matched a finite-difference estimate (error 7 × 10⁻¹¹), so the gradient was the true derivative, not just a nonzero number. A counter-example is the symmetric 0.5-initialised run: its gradients were nonzero but it stuck at 3/4 correct.
3. **Why identical/zero init prevents distinct features:** identical hidden units compute the same output and so receive the same gradient and the same update; their weights stay equal at every step (and with zero init the gradients are exactly zero). See Part C.
4. **Effect of activation on the gradient:** *Engineering observation:* the early first-layer gradient norm differed by activation (sigmoid smallest, ReLU largest in median, tanh in between), and success rates were similar across 20 seeds. *Scientific explanation:* the gradient is a product of Jacobian factors including f′(a): ≤ 0.25 for sigmoid, ≤ 1 for tanh, 1 or 0 for ReLU; sigmoid can shrink gradients, while ReLU passes them unchanged but blocks them completely when inactive. These are not proofs that one activation is better from four data points.
5. **Why output layer and loss must match the task:** the output must represent the right kind of quantity (a probability for a binary label, a probability vector for K exclusive classes) and the loss must be the negative log-likelihood of that distribution, so that the gradient is the clean p − y. Mismatches (e.g. sigmoid + squared error, or softmax for a non-exclusive multi-label task) give poorly scaled gradients or a meaningless interpretation.
7. **Which tests scale up:** keep the cheap ones: loss decreasing, correct predictions on a validation set, gradient norms and activation statistics (dead/saturated units), a few-seed repeat, and checking that outputs sum to 1. An exhaustive finite-difference gradient check needs two forward passes per parameter, so it becomes far too expensive for millions or billions of parameters; it is applied to a tiny model or to a few randomly sampled parameters instead.
