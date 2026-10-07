"""Overlap check between the train, dev and test splits of a dataset folder.

  python -m data.check_overlap --root datasets/main --raw <raw data folder> --out results/overlap_main

For each pair of the splits present (in the order train, dev, test: train-dev, train-test, dev-test) and each
criterion, counts the items of the later split that match some item of the earlier split:
  item            same state, question and options (lower-cased, trimmed)
  state           same state text (story, theory or premises), as in the build scripts
  state_norm      same state after dropping case, punctuation and extra spaces
  sentence_set    same set of sentences in any order
  near_dup        sentence sets with Jaccard overlap >= 0.8 (the maximum over the earlier split is kept)
  clutrr_renamed  CLUTRR items (dataset "clutrr", whatever file they are in): same story and question once people's
                  names are replaced by placeholders
Context, not a leak: CLUTRR items whose relation chain (f_comb, e.g. "father-sister") occurs in training.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from . import real
from .common import read_jsonl

SPLITS = ["train", "dev", "test"]
CRITERIA = ["item", "state", "state_norm", "sentence_set", "near_dup", "clutrr_renamed"]


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def sentence_set(state):
    return frozenset(n for n in (norm(p) for p in re.split(r"(?<=[.!?])\s+|\n+", state)) if n)


def load(root):
    data = {}
    for split in SPLITS:
        if not os.path.isdir(os.path.join(root, split)):
            continue
        items = []
        for p in sorted(glob.glob(os.path.join(root, split, "*.jsonl.gz"))):
            name = os.path.basename(p)[:-len(".jsonl.gz")]
            for it in read_jsonl(p):
                it["_set"] = name
                items.append(it)
        data[split] = items
    return data


def renamed(it, names):
    return real.clutrr_renamed(it, names) if it["dataset"] == "clutrr" else None


def keys(it, names):
    return {"item": (it["state"].strip().lower(), it["question"].strip().lower(), tuple(o.lower() for o in it["options"])),
            "state": it["state"].strip().lower(),
            "state_norm": norm(it["state"]),
            "sentence_set": it["_sentences"],
            "clutrr_renamed": renamed(it, names)}


def max_jaccard(later, earlier):
    """For each sentence set in `later`, the largest Jaccard overlap with any set in `earlier`, and its index."""
    post = defaultdict(list)
    for j, s in enumerate(earlier):
        for x in s:
            post[x].append(j)
    post = {x: np.asarray(v, dtype=np.int64) for x, v in post.items()}
    sizes = np.asarray([len(s) for s in earlier], dtype=np.float64)
    best, arg = np.zeros(len(later)), np.full(len(later), -1)
    for i, s in enumerate(later):
        lists = [post[x] for x in s if x in post]
        if not lists:
            continue
        inter = np.bincount(np.concatenate(lists), minlength=len(earlier))
        nz = np.nonzero(inter)[0]
        jac = inter[nz] / (len(s) + sizes[nz] - inter[nz])
        k = int(jac.argmax())
        best[i], arg[i] = jac[k], nz[k]
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the dataset folder (with train/, dev/, test/)")
    ap.add_argument("--raw", default=None, help="the raw data folder; needed when the dataset has CLUTRR items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--near", type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load(args.root)
    present = [s for s in SPLITS if s in data]
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
    has_clutrr = any(it["dataset"] == "clutrr" for items in data.values() for it in items)
    names = real.clutrr_names(args.raw) if has_clutrr else None
    for items in data.values():
        for it in items:
            it["_sentences"] = sentence_set(it["state"])
            it["_keys"] = keys(it, names)

    report, examples = {}, defaultdict(list)
    lines = ["# Overlap between splits of %s\n" % args.root,
             "Counts of items in the later split that match an item of the earlier split. "
             "near_dup = sentence-set Jaccard >= %.1f.\n" % args.near,
             "| earlier -> later | later set | n | " + " | ".join(CRITERIA) + " |",
             "|---|---|---|" + "---|" * len(CRITERIA)]
    for early, late in pairs:
        index = {c: {} for c in CRITERIA if c != "near_dup"}
        for j, it in enumerate(data[early]):
            for c, k in it["_keys"].items():
                if k is not None:
                    index[c].setdefault(k, j)
        best, arg = max_jaccard([it["_sentences"] for it in data[late]], [it["_sentences"] for it in data[early]])
        counts = defaultdict(Counter)
        for i, it in enumerate(data[late]):
            counts[it["_set"]]["n"] += 1
            hit = {c: it["_keys"][c] is not None and it["_keys"][c] in index[c] for c in index}
            hit["near_dup"] = best[i] >= args.near
            for c in CRITERIA:
                if hit[c]:
                    counts[it["_set"]][c] += 1
                    if len(examples[(early, late, c)]) < 3:
                        j = index[c][it["_keys"][c]] if c != "near_dup" else int(arg[i])
                        examples[(early, late, c)].append({"later": [it["_set"], it["state"][:300], it["question"][:200]],
                                                           "earlier": [data[early][j]["_set"], data[early][j]["state"][:300],
                                                                       data[early][j]["question"][:200]],
                                                           "jaccard": round(float(best[i]), 3)})
        report["%s->%s" % (early, late)] = {name: dict(c) for name, c in counts.items()}
        for name in sorted(counts):
            c = counts[name]
            lines.append("| %s -> %s | %s | %d | %s |" % (early, late, name, c["n"], " | ".join(str(c[x]) for x in CRITERIA)))
        report["%s->%s" % (early, late)]["_max_jaccard_hist"] = {
            ">=0.5": int((best >= 0.5).sum()), ">=0.8": int((best >= 0.8).sum()), ">=0.9": int((best >= 0.9).sum()),
            "==1": int((best >= 1 - 1e-9).sum())}

    # context: CLUTRR relation chains seen in training (expected for k <= 4, impossible for k >= 5)
    train_chains = {it["meta"]["f_comb"] for it in data["train"] if it["dataset"] == "clutrr"}
    ctx = defaultdict(Counter)
    for split in (s for s in present if s != "train"):
        for it in data[split]:
            if it["dataset"] == "clutrr":
                ctx["%s k=%d" % (split, it["k"])]["n"] += 1
                ctx["%s k=%d" % (split, it["k"])]["chain_in_train"] += it["meta"]["f_comb"] in train_chains
    report["clutrr_chain_in_train"] = {k: dict(v) for k, v in ctx.items()}
    if ctx:
        lines += ["", "CLUTRR relation chains that also occur in training (context, not a leak; the benchmark reuses "
                  "chains at k 2-4 by design):", ""]
    for k in sorted(ctx, key=lambda s: (s.split()[0], int(s.split("=")[1]))):
        lines.append("- %s: %d of %d" % (k, ctx[k]["chain_in_train"], ctx[k]["n"]))
    lines += ["", "Largest sentence-set Jaccard per later item (all sets pooled):", ""]
    for pair in ("%s->%s" % p for p in pairs):
        lines.append("- %s: %s" % (pair, report[pair]["_max_jaccard_hist"]))

    with open(os.path.join(args.out, "overlap.json"), "w") as f:
        json.dump({"report": report, "examples": {"|".join(k): v for k, v in examples.items()}}, f, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
