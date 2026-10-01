"""Shared data and helpers for the Bayesian-network language model lab."""

START, END = "<START>", "<END>"

RAW = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""


def load_sentences(raw=RAW):
    """Lower-case, split into word tokens, add <START>/<END>."""
    return [[START] + line.lower().split() + [END]
            for line in raw.strip().splitlines()]


def check_normalisation(probs, tol=1e-9):
    """Property test: for every context, sum_v P(v | context) == 1."""
    ok = True
    for ctx, dist in probs.items():
        total = sum(dist.values())
        flag = "OK " if abs(total - 1.0) < tol else "BAD"
        if flag == "BAD":
            ok = False
        print(f"  {flag} {str(ctx):28s} total = {total:.6f}")
    return ok
