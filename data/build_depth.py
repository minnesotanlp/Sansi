"""The dataset folders of the two depth tasks, from the items written by data.generate_depth.

The generated items (program-generated; their depth k is exact and their labels are computed by the generators) are
rewritten so that train.train and eval.evaluate read them. Nothing about an item changes; it only gets the fields
those scripts use.

  python -m data.build_depth --src datasets/generated_liars --out datasets/depth_liars --tokenizer <Ouro-1.4B>
  python -m data.build_depth --src datasets/generated_swaps --out datasets/depth_swaps --tokenizer <Ouro-1.4B> \
      --families shuffle

  train  the training files (k = 1..8) minus the dev items
  dev    about 7% of every family x k cell of the training files (at most 40, at least 2), chosen by a hash of the
         fingerprint, so the split is the same wherever it is built; used only for the checks during training
  test   the test files as they are (k = 1..16; rooms 2..16): tier "trained_k" for k <= 8, "deeper_k" for k > 8

Per item: qtype "noul" for the yes/no families (option order kept, as for the yes/no items of the main set) and
"choice" for the families with other options (shuffled in training as for every choice item; the test keeps the
generator's order); meta.layer = the family; meta.source_set = <family>_k<kk>, so the metrics come per family and
depth; meta.group = <family>/<fingerprint> (every item is its own group: no two items share a hidden structure);
meta.tokens = the prompt's length in Ouro tokens (SmolLM2 has the same tokenizer); the generator's own meta, with the
hidden structure and its step-by-step trace, is kept under meta.gen.
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

from data.common import read_jsonl, write_jsonl

FAMILIES = ["rooms", "liars", "ledger", "policy"]
TRAINED_MAX_K = 8


def dev_rank(fp):
    """A stable pseudo-random rank per item: the lowest ranks of a cell are its dev items."""
    return int(hashlib.sha1(("depth-dev-" + fp).encode()).hexdigest(), 16)


def convert(it, split, tier, n_tokens):
    fam = it["family"]
    out = {"dataset": "sansi_depth_v1", "family": fam, "split": split, "k": it["k"], "depth_kind": "exact_k",
           "qtype": "noul" if it["options"] == ["yes", "no"] else "choice",
           "state": it["state"], "question": it["question"], "options": it["options"],
           "answer_index": it["answer_index"], "answer_distribution": it["answer_distribution"],
           "deterministic": it["deterministic"], "fingerprint": it["fingerprint"],
           "meta": {"layer": fam, "tier": tier, "source_set": "%s_k%02d" % (fam, it["k"]),
                    "group": "%s/%s" % (fam, it["fingerprint"]), "tokens": n_tokens, "gen": it["meta"]}}
    assert it["deterministic"] and it["answer_index"] is not None, "only the deterministic items are converted"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a folder written by data.generate_depth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="the folder of Ouro-1.4B: its tokenizer counts meta.tokens")
    ap.add_argument("--dev-share", type=float, default=0.07)
    ap.add_argument("--families", default=",".join(FAMILIES), help="the families of --src to convert")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from sansi.model import render_prompt
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def n_tokens(it):
        return len(tok.encode(render_prompt(it), add_special_tokens=False))

    files, totals, seen = {}, Counter(), set()
    for fam in args.families.split(","):
        train = read_jsonl(os.path.join(args.src, "%s_train.jsonl.gz" % fam))
        test = read_jsonl(os.path.join(args.src, "%s_test.jsonl.gz" % fam))
        texts = {(it["state"], it["question"]) for it in train}
        assert not ({it["fingerprint"] for it in train} & {it["fingerprint"] for it in test})
        assert not any((it["state"], it["question"]) in texts for it in test), "a test item repeats a training text"
        cells = defaultdict(list)
        for it in train:
            cells[it["k"]].append(it)
        parts = {"train": [], "dev": [], "test": []}
        for k, items in sorted(cells.items()):
            n_dev = max(2, min(40, round(args.dev_share * len(items))))
            order = sorted(items, key=lambda it: dev_rank(it["fingerprint"]))
            dev_fps = {it["fingerprint"] for it in order[:n_dev]}
            for it in items:                                   # the files' own order is kept
                split = "dev" if it["fingerprint"] in dev_fps else "train"
                parts[split].append(convert(it, split, split, n_tokens(it)))
        for it in test:
            tier = "trained_k" if it["k"] <= TRAINED_MAX_K else "deeper_k"
            parts["test"].append(convert(it, "test", tier, n_tokens(it)))
        for split, items in parts.items():
            fps = [it["fingerprint"] for it in items]
            assert len(set(fps)) == len(fps) and not (set(fps) & seen), "an item appears twice"
            seen |= set(fps)
            path = os.path.join(args.out, split, "%s.jsonl.gz" % fam)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_jsonl(items, path)
            tk = sorted(it["meta"]["tokens"] for it in items)
            files["%s/%s" % (split, fam)] = {
                "n": len(items), "k": [min(it["k"] for it in items), max(it["k"] for it in items)],
                "per_k": dict(sorted(Counter(it["k"] for it in items).items())),
                "tokens_median": tk[len(tk) // 2], "tokens_max": tk[-1],
                "answers": dict(sorted(Counter(it["options"][it["answer_index"]] for it in items).items()))}
            totals[split] += len(items)
    manifest = {"spec": "sansi depth v1: generated depth items as a dataset folder (data/build_depth.py)",
                "args": vars(args),
                "source_manifest": json.load(open(os.path.join(args.src, "manifest.json"))).get("args"),
                "trained_max_k": TRAINED_MAX_K, "totals": dict(totals), "files": files}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("totals", dict(totals))
    for name, v in sorted(files.items()):
        print(name, "n", v["n"], "k", v["k"], "tokens median/max", v["tokens_median"], v["tokens_max"])


if __name__ == "__main__":
    main()
