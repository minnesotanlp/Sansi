"""The numbers of the paper's main table from the per-item records of one or more runs (written by eval.evaluate).

  python -m eval.report --a results/sansi_s0/test_step_1000 results/sansi_s1/test_step_1000 results/sansi_s2/test_step_1000
  python -m eval.report --a <the seeds of SanSi> --b <the seeds of a single-pass model> --loop-a 8 --loop-b 1
  python -m eval.report --a <the seeds of SanSi on a depth task> --layer liars --by k

For every recorded loop of a model: accuracy on all items and by test group (--by tier, the default), by item type
(--by layer) or by depth (--by k), then ECE, the hard-answer rate and the evidence AUROC on all items; each as the
mean over the runs (seeds) with the sample standard deviation. With --b: the difference in accuracy between model A at
--loop-a and model B at --loop-b with a 95% bootstrap interval over groups of related items, and the same difference
seed by seed when both models have the same number of runs.

  correct          an answerable item: the top option is the gold option (the majority label of a crowd-labelled
                   item); an unanswerable item: no hard answer, i.e. top probability < (1 + 1/K) / 2
  accuracy         share of items correct
  ECE              top probability against correctness in 10 equal-width bins, answerable items only
  hard answers     share of unanswerable items with top probability >= (1 + 1/K) / 2
  evidence AUROC   the chance that an answerable item gets a higher top probability than an unanswerable one (ties
                   count one half), over the items of the sources that hold both (PAIR_SOURCES)
  interval         the groups of related items (the records' "group") are drawn with replacement 2,000 times; the
                   correctness of an item is first averaged over the runs of each model; 2.5th and 97.5th percentiles

No model is run. All runs must hold records of the same items.
"""
import argparse
import gzip
import json
import os

import numpy as np

TIER_ORDER = ["in_dist", "near", "far", "jevbench", "trained_k", "deeper_k"]
PAIR_SOURCES = ["musique_pairs_id", "squad2_id", "musique_pairs_4hop", "kev_v9_unknowable_pairs"]


def load(p):
    """The records of one run as {fingerprint: record}; p is the records file or the folder that holds it."""
    if os.path.isdir(p):
        p = os.path.join(p, "records.jsonl.gz")
    with gzip.open(p, "rt") as f:
        recs = {r["fp"]: r for r in map(json.loads, f)}
    return recs


def arrays(recs, order):
    """Per item (in the given order) and loop: the top probability and whether the item is answered correctly."""
    N = len(order)
    T = len(recs[order[0]]["p"])
    top = np.zeros((N, T)); cor = np.zeros((N, T), bool)
    for i, f in enumerate(order):
        r = recs[f]; P = np.asarray(r["p"], float)
        top[i] = P.max(1)
        if r["unans"]:
            cor[i] = top[i] < (1 + 1 / P.shape[1]) / 2
        else:
            y = r["y"] if r["y"] >= 0 else int(np.argmax(r["dist"]))
            cor[i] = P.argmax(1) == y
    return {"top": top, "cor": cor}


def ece(conf, corr):
    b = np.minimum((conf * 10).astype(int), 9)
    return float(sum(abs(conf[b == i].mean() - corr[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any()))


def auroc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); o = allv.argsort(kind="mergesort"); sv = allv[o]
    rk = np.empty(len(sv)); i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        rk[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    r = np.empty(len(sv)); r[o] = rk
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else None


def sd(xs):
    """Sample standard deviation (n - 1) over the seeds; None when there is only one run."""
    xs = [x for x in xs if x == x]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def metrics(runs, mk, t, unans, pairs):
    """The metrics of the items in the mask mk at loop t (0-based): means over the runs, and their sd under "sd"."""
    m2 = mk & ~unans; mu = mk & unans; mp = mk & pairs
    out = {"n": int(mk.sum())}
    per = {"acc": [a["cor"][mk, t].mean() for a in runs]}   # one value per seed
    if m2.any():
        per["ece"] = [ece(a["top"][m2, t], a["cor"][m2, t].astype(float)) for a in runs]
    if mu.any():
        out["n_unans"] = int(mu.sum()); per["hard"] = [1 - a["cor"][mu, t].mean() for a in runs]
    if mp.any():
        out["n_pairs"] = int(mp.sum()); per["auroc_evid"] = [auroc(a["top"][mp, t], ~unans[mp]) for a in runs]
    for k, xs in per.items():
        out[k] = mean(xs)
    out["sd"] = {k: sd(xs) for k, xs in per.items()}
    return out


def boot(runs_a, ta, runs_b, tb, mk, ginv, n_groups, rng, B=2000):
    """Accuracy of model A at loop ta minus model B at loop tb (0-based) on the items in mk: the difference and its
    95% interval, resampling the groups of related items."""
    da = np.mean([x["cor"][:, ta].astype(float) for x in runs_a], 0); db = np.mean([x["cor"][:, tb].astype(float) for x in runs_b], 0)
    d = (da - db) * mk
    gs = np.bincount(ginv, weights=d, minlength=n_groups); gn = np.bincount(ginv, weights=mk.astype(float), minlength=n_groups)
    keep = gn > 0; gs, gn = gs[keep], gn[keep]
    idx = rng.integers(0, len(gs), size=(B, len(gs)))
    v = gs[idx].sum(1) / gn[idx].sum(1)
    return [float(gs.sum() / gn.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def masks(by, ref, order, sel):
    """The item groups to report, {name: mask}, "all" first; sel restricts every group (--layer)."""
    if by == "k":
        key = np.array([-1 if ref[f]["k"] is None else ref[f]["k"] for f in order])
        names = [int(k) for k in sorted(set(key[sel])) if k >= 0]
    else:
        key = np.array([ref[f][by] for f in order])
        names = sorted(set(key[sel]))
        if by == "tier":
            names = [n for n in TIER_ORDER if n in names] + [n for n in names if n not in TIER_ORDER]
    out = {"all": sel.copy()}
    for n in names:
        out["k=%d" % n if by == "k" else str(n)] = (key == n) & sel
    return out


def cell(x, s, scale=1.0, fmt="%.1f"):
    if x is None:
        return "-"
    return fmt % (scale * x) + ("" if s is None else (" ± " + fmt) % (scale * s))


def table(title, runs, groups, unans, pairs):
    T = runs[0]["top"].shape[1]
    head = ["loop"] + ["%s (%d)" % (g, mk.sum()) for g, mk in groups.items()] + ["ECE", "hard answers", "evidence AUROC"]
    lines = [title, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in range(T):
        a = metrics(runs, groups["all"], t, unans, pairs)
        row = [str(t + 1)] + [cell(m["acc"], m["sd"]["acc"], 100) for m in (metrics(runs, mk, t, unans, pairs)
                                                                         for mk in groups.values())]
        row += [cell(a.get("ece"), a["sd"].get("ece"), 1, "%.3f"), cell(a.get("hard"), a["sd"].get("hard"), 100),
                cell(a.get("auroc_evid"), a["sd"].get("auroc_evid"), 1, "%.3f")]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="model A: the records of its runs (one per seed); each a "
                    "folder written by eval.evaluate or its records.jsonl.gz")
    ap.add_argument("--b", nargs="+", default=None, help="model B, for the difference A - B")
    ap.add_argument("--loop-a", type=int, default=0, help="the loop of A in the difference (default: its last)")
    ap.add_argument("--loop-b", type=int, default=0, help="the loop of B in the difference (default: its last)")
    ap.add_argument("--by", default="tier", choices=["tier", "layer", "k"], help="accuracy by test group (tier), by "
                    "item type (layer) or by depth (k)")
    ap.add_argument("--layer", default=None, help="only the items of this item type, e.g. liars or shuffle for the "
                    "depth tasks")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
    args = ap.parse_args()

    data = {"A": [load(p) for p in args.a]}
    if args.b:
        data["B"] = [load(p) for p in args.b]
    ref = data["A"][0]
    order = list(ref)
    N = len(order)
    for name, runs in data.items():
        for r in runs:
            assert len(r) == N and all(f in r for f in order), "model %s: the runs do not hold the same items" % name
            assert len({len(x["p"]) for x in r.values()}) == 1, "model %s: items with different numbers of loops" % name
        assert len({len(r[order[0]]["p"]) for r in runs}) == 1, "model %s: runs with different numbers of loops" % name
    unans = np.array([ref[f]["unans"] for f in order]); soft = np.array([ref[f]["soft"] for f in order])
    group = np.array([ref[f]["group"] for f in order]); source = np.array([ref[f]["source"] for f in order])
    layer = np.array([ref[f]["layer"] for f in order])
    ug, ginv = np.unique(group, return_inverse=True)
    pairs = np.isin(source, PAIR_SOURCES) & ~soft
    sel = np.ones(N, bool) if args.layer is None else layer == args.layer
    assert sel.any(), "no item of type %s" % args.layer
    groups = masks(args.by, ref, order, sel)
    A = {name: [arrays(r, order) for r in runs] for name, runs in data.items()}

    what = {"tier": "test group", "layer": "item type", "k": "depth"}[args.by]
    for name, runs in A.items():
        print(table("Model %s: %d run%s, %d items%s. Accuracy (%%) by %s (number of items); ECE, hard answers (%%) and "
                    "evidence AUROC on all of them. Mean%s." % (
                        name, len(runs), "" if len(runs) == 1 else "s", int(sel.sum()),
                        "" if args.layer is None else " of type %s" % args.layer, what,
                        "" if len(runs) == 1 else " ± standard deviation over the runs"),
                    runs, groups, unans, pairs))
    if args.b:
        ta = (args.loop_a or A["A"][0]["top"].shape[1]) - 1
        tb = (args.loop_b or A["B"][0]["top"].shape[1]) - 1
        rng = np.random.default_rng(args.seed)
        paired = len(A["A"]) == len(A["B"])
        print("Accuracy of A at loop %d minus B at loop %d, in points, with the 95%% bootstrap interval over groups of "
              "related items (%d resamples)%s.\n" % (ta + 1, tb + 1, args.resamples,
                                                     "; by seed: run i of A minus run i of B" if paired else ""))
        print("| %s | items | A - B | 95%% interval |%s" % (what, " by seed |" if paired else ""))
        print("|---|---|---|---|" + ("---|" if paired else ""))
        for g, mk in groups.items():
            d, lo, hi = boot(A["A"], ta, A["B"], tb, mk, ginv, len(ug), rng, args.resamples)
            row = "| %s | %d | %+.1f | [%+.1f, %+.1f] |" % (g, mk.sum(), 100 * d, 100 * lo, 100 * hi)
            if paired:
                row += " %s |" % " ".join("%+.1f" % (100 * float(x["cor"][mk, ta].mean() - y["cor"][mk, tb].mean()))
                                          for x, y in zip(A["A"], A["B"]))
            print(row)


if __name__ == "__main__":
    main()
