"""Greedy vs sampling, and first-order vs second-order comparison."""
import random
from common import START, END, load_sentences
from first_order import FirstOrderLM
from second_order import SecondOrderLM

sents = load_sentences()
train = {" ".join(s[1:-1]) for s in sents}
m1, m2 = FirstOrderLM(sents), SecondOrderLM(sents)

print("=== Part VIII: argmax next word (first-order) ===")
for w in ["the", "cat", "dog", "sat", "ran", "on", "to"]:
    print(f"  argmax P(. | {w:4s}) = {m1.predict(w)}   {m1.distribution(w)}")

print("\n=== Part X: greedy vs sampling (5 each) ===")
rng = random.Random(1)
for name, m in [("first-order", m1), ("second-order", m2)]:
    print(f"\n{name}:")
    print("  greedy  :", [m.generate("greedy") for _ in range(5)])
    samp = [m.generate("sample", rng) for _ in range(5)]
    for s in samp: print("  sample  :", s)

print("\n=== Part XIII: model comparison (1000 samples each) ===")
V = len(m1.vocab)
nxt_vocab = V - 1                      # <START> is never predicted
ctx1_possible = V - 1                  # <END> is never a context
ctx2_possible = (V - 1) ** 2           # any pair of non-END tokens (incl. <START>)
for name, m, ctxp in [("first-order", m1, ctx1_possible), ("second-order", m2, ctx2_possible)]:
    rng = random.Random(42)
    gen = [m.generate("sample", rng) for _ in range(1000)]
    distinct = set(gen)
    novel = distinct - train
    observed_ctx = len(m.probs)
    print(f"{name}:")
    print(f"  non-zero parameters (distinct observed transitions): {m.n_parameters()}")
    print(f"  full CPT size (contexts x next tokens): {ctxp} x {nxt_vocab} = {ctxp*nxt_vocab}")
    print(f"  observed contexts: {observed_ctx}/{ctxp}  -> unseen contexts: {ctxp-observed_ctx}")
    print(f"  distinct sentences in 1000 samples: {len(distinct)}; of which not in training data: {len(novel)}")
    print(f"  novel examples: {sorted(novel)[:6]}")
print("\nVocab size incl. START/END:", V, m1.vocab)
print("Distinct training sentences:", len(train))
