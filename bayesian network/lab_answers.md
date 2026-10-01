# AI Laboratory: Bayesian Networks and Autoregressive Language Models
*Answers and results. All numbers come from running `first_order.py`, `second_order.py`, `compare.py`.*

## Files
| File | Purpose |
|---|---|
| `common.py` | dataset, `<START>/<END>` handling, normalisation test |
| `first_order.py` | first-order model P(Xt \| Xt-1) |
| `second_order.py` | second-order model P(Xt \| Xt-2, Xt-1) |
| `compare.py` | argmax tables, greedy vs sampling, model comparison |
| `generated_first_order.txt`, `generated_second_order.txt` | 20 generated sentences each |

---
## Q1. Why is the chain-rule decomposition useful for generating text?
It turns one huge problem (the probability of a whole sentence) into a sequence of small ones: "given what I have so far, what comes next?" Each factor P(xt | x1..xt-1) is a distribution over one token, so we can generate by sampling x1, appending it, sampling x2 given x1, and so on. The decomposition is exact (no assumption), so generating this way samples from the joint distribution.

## Q2. Independence assumption of X1 → X2 → X3 → ... → XT
Each word depends only on the previous word. Given Xt-1, Xt is conditionally independent of everything earlier:

P(Xt | X1, ..., Xt-1) = P(Xt | Xt-1), i.e. Xt ⟂ (X1..Xt-2) | Xt-1.

This is the Markov property.

## Q3. CPT P(next | current) (counts from the 6 training sentences)

| current | next word : probability |
|---|---|
| `<START>` | the : 1.000 |
| **the** | cat : 3/12 = 0.250, dog : 3/12 = 0.250, mat : 2/12 = 0.167, rug : 2/12 = 0.167, park : 2/12 = 0.167 |
| **cat** | sat : 2/3 = 0.667, ran : 1/3 = 0.333 |
| **dog** | sat : 2/3 = 0.667, ran : 1/3 = 0.333 |
| **sat** | on : 1.000 |
| **ran** | to : 1.000 |
| on | the : 1.000 |
| to | the : 1.000 |
| mat / rug / park | `<END>` : 1.000 |

**Zero-probability transitions** (examples): the → {on, ran, sat, the, to, `<END>`}; cat → {cat, dog, the, on, mat, ...}; sat → everything except "on"; ran → everything except "to". In particular, nothing except mat/rug/park can be followed by `<END>`, and "cat → ate"-style sentences are impossible because "ate" is not in the vocabulary at all. Zeros mean "never observed", not "impossible in English".

## Q4. Where are transition counts stored?
In `FirstOrderLM.__init__`, in `self.counts`, a `defaultdict(Counter)` where `counts[prev][next] = C(prev, next)`, filled by looping over `zip(sent[:-1], sent[1:])`.

## Q5. Where is P(Xt | Xt-1) computed?
In `__init__`, in the dictionary comprehension that builds `self.probs`: `c / sum(ctr.values())`, i.e. C(wi, wj) / Σk C(wi, wk).

## Q6. How does the program choose the next word?
`generate(mode="sample")` samples with `rng.choices(words, weights=probs)`; `generate(mode="greedy")` takes `argmax`. Greedy always returns the single most probable word, so it is deterministic: the same context always gives the same output. Sampling picks word w with probability P(w | context), so likely words are chosen more often but unlikely words still appear. Only sampling draws from the model's distribution; greedy ignores everything except the mode.

## Q7. What happens with an unseen context?
`self.probs.get(prev, {})` returns an empty distribution; `predict` returns `None` and `sample_next` returns `<END>`, so generation stops. (Without this guard the code would raise a `KeyError`.) This is a design choice; real systems use smoothing (e.g. add-one) or back-off to a shorter context so that every context has a distribution.

## Q8. A total of 0.87
The CPT row is not a valid probability distribution: 13% of the probability mass is missing. Likely causes: dividing by the wrong denominator (e.g. total counts for a different word, or omitting `<END>` transitions from the sum), dropping some transitions, or a rounding/filtering bug. Sampling from it would be biased.

**Normalisation test results:** every context in both models sums to 1.000000 (see program output).

## Q9. Do predictions match human expectation?
Argmax predictions (first-order):

| context | argmax | note |
|---|---|---|
| the | cat (tie with dog at 0.25) | tie, broken alphabetically |
| cat | sat (0.667) | matches intuition |
| dog | sat (0.667) | matches |
| sat | on (1.0) | matches |
| ran | to (1.0) | matches |
| on | the (1.0) | matches |
| to | the (1.0) | matches |

Not always. After "the cat sat on the", a human expects "mat" or "rug", but the first-order model only sees "the" and predicts cat/dog (0.25 each), which gives ungrammatical-looking sentences such as "the dog sat on the dog ran to the park". A probability model reflects only the statistics of its training data and the context it is allowed to see; human expectations use world knowledge, grammar and longer context.

## Q10. Greedy vs sampling
- **Greedy**: no variation. First-order greedy produced the same output 5/5 times and got stuck in a cycle: `the cat sat on the cat sat on the ...` (the → cat → sat → on → the → ...), only stopping at my `max_len` safeguard. Second-order greedy gave `the cat sat on the mat` 5/5 times.
- **Sampling**: first-order gave the park / the rug / the dog sat on the rug ...; second-order gave a mix of the training sentences.

Sampling varies more because it draws from the whole distribution; greedy always collapses to the mode (and ties are resolved by an arbitrary rule).

## Q11. First- vs second-order
1. **Graph:** first-order has one parent per node (Xt-1 → Xt, a chain); second-order has two parents (Xt-2 → Xt ← Xt-1), adding skip edges.
2. **CPT:** first-order has |V| contexts, so up to |V|×|V| entries; second-order has |V|² contexts, up to |V|³ entries. (Here V=12 incl. START/END: 11×11=121 vs 121×11=1331 entries.)
3. **Context:** two previous tokens instead of one: the model can distinguish "on the → mat/rug" from "to the → park".
4. **Data:** much more. Each context has fewer observations, and most contexts are never seen (106 of 121 here).

## Q12. More context: better prediction, harder estimation
More context removes ambiguity (e.g. "the" after "on" vs after "to"), so P(next | longer context) can be much sharper and closer to the true distribution. But the CPT grows exponentially, |V|^(k+1) entries for order k, while the data stays fixed. Our results: first-order has 17 non-zero parameters from 121 possible cells and all 11 contexts observed; second-order has 19 non-zero parameters from 1331 cells and only 15 of 121 contexts observed. Contexts with 1-2 observations give unreliable estimates (the model memorises), and unseen contexts give no prediction at all.

## Part XIII: comparison (1000 sampled sentences each)
| | first-order | second-order |
|---|---|---|
| non-zero parameters | 17 | 19 |
| full CPT cells | 121 | 1331 |
| unseen contexts | 0 / 11 | 106 / 121 |
| distinct sentences in 1000 samples | 167 (161 not in training data) | 6 (0 novel) |

*Examples.* First-order: "the cat sat on the park", "the park", "the cat ran to the dog sat on the rug": varied but often incoherent or wrongly ended. Second-order: "the dog ran to the park", "the cat sat on the rug": all fluent but exactly the 6 training sentences. So the second-order model is coherent but has memorised the data, while the first-order model generalises more but is less coherent. That is the bias/variance trade-off in miniature.

## Q13. Why is Approach B preferable?
- **Specifying the behaviour:** B states the model (P(Xt|Xt-1), counts, sampling), so there is something definite to check the code against; A lets the LLM pick any design (it might return a neural net or a rule-based generator).
- **Understanding the representation:** you know where counts and CPTs should live, so you can find them in the code (Q4/Q5).
- **Validating the implementation:** you can compare the CPT with your hand-computed one from Q3 (e.g. P(cat|the) = 0.25).
- **Testing invariants:** rows must sum to 1; probabilities in [0,1]; unseen contexts handled; generation stops at `<END>`.
- **Implementation vs model:** the same model can be coded many ways; a program that "runs and prints sentences" is not necessarily a correct implementation of the intended probability model. Only an explicit specification lets you tell the difference.

## Q14. What did the Bayesian-network view give us?
- **Factorisation of the joint:** P(X1..XT) = ΠP(Xt | parents(Xt)). The chain rule says what to estimate (one CPT per node), and the graph says which variables each CPT conditions on.
- **Reasoning about independence assumptions:** the first-order graph says Xt ⟂ X1..Xt-2 | Xt-1, and the failure "the dog sat on the dog ..." is exactly the cost of that assumption.
- **Effect of increasing context:** adding the edge Xt-2 → Xt enlarges the CPT from |V|² to |V|³, which explains both the better coherence and the data sparsity.
- **Testing against the specification:** each CPT row must be a distribution, so normalisation is a test that the code matches the model.
- **Principled generation:** ancestral sampling, from parents to children, samples from the joint.
