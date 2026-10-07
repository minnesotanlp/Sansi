"""Generate the items of the depth tasks: generate, verify, de-duplicate, probe for shortcuts, write JSONL.

  python -m data.generate_depth --out datasets/generated_liars --probe
  python -m data.generate_depth --out datasets/generated_swaps --families shuffle --probe

Writes {family}_train.jsonl.gz, {family}_test.jsonl.gz and manifest.json (plain .jsonl with --no-gzip). Train and
test are disjoint by structure fingerprint. Generation is deterministic given --seed. data.build_depth then turns a
generated folder into a dataset folder that the training and test scripts read (see data/README.md).
"""
import argparse
import json
import os
import statistics
import time

from . import ledger, liars, policy, probe, rooms, shuffle
from .common import approx_tokens, dedup, write_jsonl

FAMILIES = {"rooms": rooms, "ledger": ledger, "liars": liars, "policy": policy, "shuffle": shuffle}


def krange(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def summarize(items):
    ks = sorted({it["k"] for it in items})
    toks = [approx_tokens(it["state"] + " " + it["question"]) for it in items]
    return {"n": len(items), "k": [ks[0], ks[-1]], "tokens_mean": round(statistics.mean(toks)),
            "tokens_max": max(toks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="rooms,ledger,liars,policy")
    ap.add_argument("--train-k", default="1-8")
    ap.add_argument("--test-k", default="1-16")
    ap.add_argument("--per-cell-train", type=int, default=600)
    ap.add_argument("--per-cell-test", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--no-gzip", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ext = ".jsonl" if args.no_gzip else ".jsonl.gz"
    manifest = {"args": vars(args), "families": {}}
    for name in args.families.split(","):
        mod = FAMILIES[name]
        t0 = time.time()
        n_opt = 3 if name == "policy" else 2
        pct_train = args.per_cell_train - args.per_cell_train % n_opt
        pct_test = args.per_cell_test - args.per_cell_test % n_opt
        train = mod.generate(krange(args.train_k), pct_train, seed=args.seed * 1000 + 1)
        test = mod.generate(krange(args.test_k), pct_test, seed=args.seed * 1000 + 2)
        n_train_raw, n_test_raw = len(train), len(test)
        train = dedup(train)
        test = dedup(test, seen={it["fingerprint"] for it in train})
        for it in train + test:
            assert mod.verify(it), it["fingerprint"]
        write_jsonl(train, os.path.join(args.out, f"{name}_train{ext}"))
        write_jsonl(test, os.path.join(args.out, f"{name}_test{ext}"))
        rec = {"train": summarize(train), "test": summarize(test),
               "dropped_duplicates": {"train": n_train_raw - len(train), "test": n_test_raw - len(test)}}
        if args.probe:
            rec["probe"] = probe.run(train, test, mod.HEURISTICS)
        rec["seconds"] = round(time.time() - t0, 1)
        manifest["families"][name] = rec
        print(name, json.dumps(rec, indent=1))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
