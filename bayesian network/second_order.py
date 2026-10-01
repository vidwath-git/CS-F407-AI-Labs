"""Second-order autoregressive LM:  P(X_t | X_{t-2}, X_{t-1})."""
import random
from collections import defaultdict, Counter
from common import START, END, load_sentences, check_normalisation


class SecondOrderLM:
    def __init__(self, sentences):
        # counts of observed triples, grouped by the 2-token context
        self.counts = defaultdict(Counter)
        for sent in sentences:
            padded = [START] + sent          # two <START>s so P(X1 | <S>,<S>) works
            for a, b, c in zip(padded, padded[1:], padded[2:]):
                self.counts[(a, b)][c] += 1
        self.probs = {
            ctx: {w: c / sum(ctr.values()) for w, c in ctr.items()}
            for ctx, ctr in self.counts.items()
        }
        self.vocab = sorted({t for s in sentences for t in s})

    def distribution(self, ctx):
        return self.probs.get(ctx, {})

    def show(self, ctx):
        dist = self.distribution(ctx)
        print(f"P(next | {ctx[0]}, {ctx[1]}):")
        for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"   {w:8s} {p:.3f}")

    def predict(self, ctx):
        dist = self.distribution(ctx)
        return max(sorted(dist), key=lambda w: dist[w]) if dist else None

    def sample_next(self, ctx, rng):
        dist = self.distribution(ctx)
        if not dist:
            return END
        words, weights = zip(*dist.items())
        return rng.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sample", rng=None, max_len=20):
        rng = rng or random
        ctx, out = (START, START), []
        while len(out) < max_len:
            nxt = self.predict(ctx) if mode == "greedy" else self.sample_next(ctx, rng)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            ctx = (ctx[1], nxt)
        return " ".join(out)

    def n_parameters(self):
        return sum(len(d) for d in self.probs.values())


if __name__ == "__main__":
    lm = SecondOrderLM(load_sentences())
    for ctx in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("sat", "on"), ("on", "the"), ("to", "the")]:
        lm.show(ctx); print()
    print("Normalisation test:"); print("all OK" if check_normalisation(lm.probs) else "FAILED")
    rng = random.Random(0)
    sampled = [lm.generate("sample", rng) for _ in range(20)]
    print("\n20 sampled sentences:")
    for s in sampled: print("  ", s)
    open("generated_second_order.txt", "w").write("\n".join(sampled) + "\n")
