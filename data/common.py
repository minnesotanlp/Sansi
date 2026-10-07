"""Shared pieces of the data scripts: the item fields of the program-generated tasks, reading and writing JSON lines.

Every generator of a depth task (liars, shuffle, rooms, ledger, policy) has the same three layers:

  1. sample a hidden structure whose depth k is a parameter,
  2. compute the answer from that structure with a program,
  3. render the structure as text.

No model is called anywhere in data/. Generated items share one schema:

  family, k, state, question, options, answer_index, answer_distribution, deterministic, fingerprint, meta

`meta` keeps the hidden structure and a step-by-step trace of the computation.
"""
import gzip
import hashlib
import json
import random

# Short, common first names; used where a puzzle needs people.
FIRST_NAMES = [
    "Ana", "Ben", "Cal", "Dee", "Eli", "Fay", "Gus", "Ida", "Jon", "Kim",
    "Lee", "Mia", "Ned", "Ola", "Pam", "Quin", "Ray", "Sue", "Tom", "Uma",
    "Vic", "Wes", "Yara", "Zed", "Abe", "Bea", "Cy", "Dan", "Eva", "Finn",
    "Gwen", "Hal", "Iris", "Jay", "Kai", "Liv", "Max", "Nia", "Otis", "Pia",
    "Rex", "Sam", "Tess", "Ugo", "Val", "Will", "Xena", "Yul", "Zoe", "Amy",
    "Bo", "Cleo", "Dov", "Ege", "Flo", "Gil", "Hana", "Ike", "Jill", "Kurt",
    "Lars", "Meg", "Nate", "Omar", "Pat", "Rosa", "Seth", "Tara", "Ulla", "Vera",
    "Wade", "Yves", "Zara", "Alix", "Bert", "Cora", "Dale", "Erin", "Fred", "Gail",
]


def fingerprint(*parts):
    """Stable id of the hidden structure (not of the wording), used for de-duplication."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def approx_tokens(text):
    """Rough token estimate for length budgeting (about 3.6 characters per token)."""
    return int(len(text) / 3.6) + 1


def make_item(family, k, state, question, options, answer_distribution, meta,
              deterministic=True, fp=None):
    item = {
        "family": family,
        "k": k,
        "state": state,
        "question": question,
        "options": list(options),
        "answer_distribution": [round(p, 6) for p in answer_distribution],
        "deterministic": deterministic,
        "fingerprint": fp,
        "meta": meta,
    }
    if deterministic:
        idx = max(range(len(options)), key=lambda i: answer_distribution[i])
        assert answer_distribution[idx] == 1.0, "deterministic items must have a one-hot distribution"
        item["answer_index"] = idx
    return item


def balanced(make, ks, per_cell, seed, n_options):
    """Call make(k, want, rng) so that every (k, answer) cell has the same number of items.

    `make` must return an item whose answer_index == want. per_cell must be a
    multiple of n_options so the cells are exactly balanced.
    """
    assert per_cell % n_options == 0, "per_cell must be a multiple of the number of options"
    rng = random.Random(seed)
    out = []
    for k in ks:
        for j in range(per_cell):
            want = j % n_options
            item = make(k, want, rng)
            assert item["answer_index"] == want, (item["family"], k, want, item["answer_index"])
            out.append(item)
    rng.shuffle(out)
    return out


def dedup(items, seen=None):
    """Drop items whose structure fingerprint was already seen (within items, and in `seen`)."""
    seen = set() if seen is None else set(seen)
    out = []
    for it in items:
        if it["fingerprint"] in seen:
            continue
        seen.add(it["fingerprint"])
        out.append(it)
    return out


def _open(path, mode):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(items, path):
    """Writes plain JSON lines, gzip-compressed when the path ends with .gz."""
    with _open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with _open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def cap(s):
    return s[0].upper() + s[1:]
