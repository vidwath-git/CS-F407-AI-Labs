"""First-order autoregressive LM:  P(X_t | X_{t-1})  (a Markov chain / chain-shaped BN)."""
import random
from collections import defaultdict, Counter
from common import START, END, load_sentences, check_normalisation


class FirstOrderLM:
    def __init__(self, sentences):
        # Q4: transition counts are stored here: counts[prev][next] = C(prev, next)
        self.counts = defaultdict(Counter)
        for sent in sentences:
            for prev, nxt in zip(sent[:-1], sent[1:]):
                self.counts[prev][nxt] += 1
        # Q5: P(next | prev) = C(prev, next) / sum_k C(prev, k) is computed here
        self.probs = {
            prev: {w: c / sum(ctr.values()) for w, c in ctr.items()}
            for prev, ctr in self.counts.items()
        }
        self.vocab = sorted({t for s in sentences for t in s})

    def distribution(self, prev):
        """Return P(. | prev) as a dict, or {} if prev was never seen as a context."""
        return self.probs.get(prev, {})

    def show(self, prev):
        dist = self.distribution(prev)
        print(f"P(next | {prev}):")
        for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"   {w:8s} {p:.3f}")
        zeros = [w for w in self.vocab if w not in dist and w != START]
        print(f"   zero-probability: {zeros}")

    def predict(self, prev):
        """Most probable next token (ties broken alphabetically so it is deterministic)."""
        dist = self.distribution(prev)
        if not dist:
            return None
        return max(sorted(dist), key=lambda w: dist[w])

    def sample_next(self, prev, rng):
        dist = self.distribution(prev)
        if not dist:                 # Q7: unseen context -> we cannot continue; stop.
            return END
        words, weights = zip(*dist.items())
        return rng.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sample", rng=None, max_len=20):
        rng = rng or random
        prev, out = START, []
        while len(out) < max_len:
            nxt = self.predict(prev) if mode == "greedy" else self.sample_next(prev, rng)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            prev = nxt
        return " ".join(out)

    def n_parameters(self):
        return sum(len(d) for d in self.probs.values())   # non-zero CPT entries


if __name__ == "__main__":
    lm = FirstOrderLM(load_sentences())
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to"]:
        lm.show(w); print()
    print("Normalisation test:"); print("all OK" if check_normalisation(lm.probs) else "FAILED")
    rng = random.Random(0)
    print("\n20 sampled sentences:")
    sampled = [lm.generate("sample", rng) for _ in range(20)]
    for s in sampled: print("  ", s)
    open("generated_first_order.txt", "w").write("\n".join(sampled) + "\n")
