"""Check that a built dataset folder is the paper's, file by file.

  python -m data.verify --root datasets/main --dataset main
  python -m data.verify --root datasets/depth_liars --dataset depth_liars
  python -m data.verify --root datasets/depth_swaps --dataset depth_swaps

`data/checksums.json` holds the sha256 of the uncompressed content of every file of the three datasets of the
paper and its number of items. The check passes only when the folder has exactly these files with exactly this
content; it prints every file that is missing, extra or different. The manifest is not compared, because it records
the arguments of the build (local paths).
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

CHECKSUMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksums.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a dataset folder written by data.build_main or data.build_depth")
    ap.add_argument("--dataset", required=True, choices=["main", "depth_liars", "depth_swaps"])
    args = ap.parse_args()
    want = json.load(open(CHECKSUMS))[args.dataset]["files"]
    have = {os.path.relpath(f, args.root) for f in glob.glob(os.path.join(args.root, "*", "*.jsonl.gz"))}
    bad = []
    for rel in sorted(set(want) | have):
        if rel not in have:
            bad.append((rel, "missing"))
            continue
        if rel not in want:
            bad.append((rel, "not in the paper's dataset"))
            continue
        content = gzip.open(os.path.join(args.root, rel)).read()
        if hashlib.sha256(content).hexdigest() != want[rel]["sha256"]:
            bad.append((rel, "different content (%d items, the paper's file has %d)" % (content.count(b"\n"),
                                                                                       want[rel]["items"])))
    if bad:
        print("%s is not the paper's %s dataset; %d of %d files differ:" % (args.root, args.dataset, len(bad),
                                                                          len(want)))
        for rel, why in bad:
            print("  %s: %s" % (rel, why))
        sys.exit(1)
    print("%s is identical to the paper's %s dataset: %d files, %d items" % (
        args.root, args.dataset, len(want), sum(v["items"] for v in want.values())))


if __name__ == "__main__":
    main()
