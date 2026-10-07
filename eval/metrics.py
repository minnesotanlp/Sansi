"""Metrics of one run's evaluation records (written by eval.evaluate.run_records): metrics.json and summary.md.

A record holds one item's option probabilities after every loop (p[t]) and its target (y = the gold option or the
majority label of a soft item, -1 for an unanswerable item; dist = the target distribution).

  correct at loop t   y >= 0: the top option is y. y = -1: not a hard answer, top probability < (1 + 1/K) / 2.
  accuracy            share of items correct (unanswerable items by the rule above)
  ECE                 top probability against correctness, 10 equal-width bins; items with a correct answer only
  Brier               sum over options of (p - target)^2; items with a correct answer only
  confident errors    share of items with a correct answer that are wrong with top probability >= 0.8
  hard answers        share of unanswerable items answered with top probability >= (1 + 1/K) / 2
  TVD                 total variation distance between p and the human (or exact) distribution; soft items only

eval.report computes the numbers of the paper's tables from the records of several runs (seeds).
"""
import math
from collections import defaultdict


def ece(probs, correct, n_bins=10):
    """Expected calibration error of the top-probability prediction."""
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    for p, c in zip(probs, correct):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b][0] += 1
        bins[b][1] += p
        bins[b][2] += c
    n = len(probs)
    return sum(cnt / n * abs(sp / cnt - sc / cnt) for cnt, sp, sc in bins.values()) if n else float("nan")


def argmax(p):
    return max(range(len(p)), key=lambda i: p[i])


def correct(rec, t):
    p = rec["p"][t]
    if rec["y"] >= 0:
        return argmax(p) == rec["y"]
    return max(p) < (1.0 + 1.0 / len(p)) / 2.0


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def by_loop(recs):
    """Per loop 1..T: the metrics above over recs (all with the same number of loops)."""
    T = len(recs[0]["p"])
    out = {}
    ans = [r for r in recs if r["y"] >= 0]
    un = [r for r in recs if r["y"] < 0]
    soft = [r for r in ans if r["soft"]]
    for t in range(T):
        maxp_a = [max(r["p"][t]) for r in ans]
        corr_a = [correct(r, t) for r in ans]
        out[t + 1] = {
            "n": len(recs),
            "accuracy": _mean(correct(r, t) for r in recs),
            "ece": ece(maxp_a, corr_a) if ans else float("nan"),
            "brier": _mean(sum((pi - di) ** 2 for pi, di in zip(r["p"][t], r["dist"])) for r in ans),
            "mean_max_prob": _mean(max(r["p"][t]) for r in recs),
            "confident_error_rate": _mean(m >= 0.8 and not c for m, c in zip(maxp_a, corr_a)),
            "hard_answer_rate": _mean(not correct(r, t) for r in un),
            "tvd": _mean(0.5 * sum(abs(pi - di) for pi, di in zip(r["p"][t], r["dist"])) for r in soft),
        }
    return out


def groups(recs):
    """Named subsets: all, per tier, per layer x tier, per source."""
    g = defaultdict(list)
    for r in recs:
        g["all"].append(r)
        g["tier/" + r["tier"]].append(r)
        g["layer/%s/%s" % (r["layer"], r["tier"])].append(r)
        g["source/" + r["source"]].append(r)
    return dict(sorted(g.items()))


def compute(recs):
    out = {}
    for name, rs in groups(recs).items():
        m = {"n": len(rs), "by_loop": by_loop(rs)}
        out[name] = m
    return out


def _f(x, d=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else ("%.*f" % (d, x))


def summary(metrics, title):
    """Markdown tables: per layer x tier at every loop (accuracy, ECE), unanswerable hard answers, TVD."""
    rows = [k for k in metrics if k.startswith("layer/") or k.startswith("tier/") or k == "all"]
    T = max(int(t) for t in metrics["all"]["by_loop"])
    loops = list(range(1, T + 1))
    lines = ["# %s" % title, ""]
    for key, label in (("accuracy", "Accuracy"), ("ece", "ECE (items with a correct answer)"),
                       ("hard_answer_rate", "Hard answers on unanswerable items"), ("tvd", "TVD on soft items")):
        lines += ["## %s by loop" % label, "", "| group | n | " + " | ".join("t%d" % t for t in loops) + " |",
                  "|---|---|" + "---|" * len(loops)]
        for k in rows:
            bl = metrics[k]["by_loop"]
            vals = [bl[t][key] if t in bl else bl[str(t)][key] for t in loops]
            if all(isinstance(v, float) and math.isnan(v) for v in vals):
                continue
            lines.append("| %s | %d | %s |" % (k, metrics[k]["n"], " | ".join(_f(v) for v in vals)))
        lines.append("")
    return "\n".join(lines) + "\n"
