"""Family `liars`: does the last person in a chain of claims tell the truth?  (flip chain)

Structure: a chain of k+1 people. The root's status is stated outright ("Ana always
tells the truth"). Each later person makes a claim about the previous one ("Ben says
Ana lies"). A claim is true iff the claimant tells the truth, so each link flips or
keeps the value: the answer is a parity over the k claims.

A decoy chain of the same length, with its own declared root, is mixed in, and all
claims are listed in random order. So the model must first find which claims lead
from the target back to a root, then resolve them. Because the decoy claims are
random, the total number of "lies" words in the text is independent of the answer.
"""
import re

from .common import FIRST_NAMES, approx_tokens, balanced, fingerprint, make_item

ROOT = {True: ["{a} always tells the truth.", "{a} is honest."],
        False: ["{a} always lies.", "{a} is a liar."]}
CLAIM = {True: ["{a} says that {b} tells the truth.", "{a} says {b} is honest.",
                "According to {a}, {b} tells the truth."],
         False: ["{a} says that {b} lies.", "{a} says {b} is a liar.",
                 "According to {a}, {b} lies."]}
QUESTIONS = ["Does {t} tell the truth?", "Is {t} telling the truth?"]


def resolve(root_value, claims):
    """claims: list of booleans (claim i says 'previous person tells the truth' iff True)."""
    vals = [root_value]
    for c in claims:
        vals.append(c == vals[-1])
    return vals


def make(k, want, rng):
    """One item with answer_index == want (0 = yes, the target tells the truth)."""
    people = rng.sample(FIRST_NAMES, 2 * (k + 1))
    chain, decoy = people[: k + 1], people[k + 1:]
    claims = [rng.random() < 0.5 for _ in range(k)]
    root = rng.random() < 0.5
    vals = resolve(root, claims)
    if vals[-1] != (want == 0):                    # flip the root: flips every value on the chain
        root = not root
        vals = resolve(root, claims)
    assert vals[-1] == (want == 0)
    d_claims = [rng.random() < 0.5 for _ in range(k)]
    d_root = rng.random() < 0.5
    d_vals = resolve(d_root, d_claims)
    lines = [rng.choice(CLAIM[c]).format(a=chain[i + 1], b=chain[i]) for i, c in enumerate(claims)]
    lines += [rng.choice(CLAIM[c]).format(a=decoy[i + 1], b=decoy[i]) for i, c in enumerate(d_claims)]
    rng.shuffle(lines)
    roots = [rng.choice(ROOT[root]).format(a=chain[0]), rng.choice(ROOT[d_root]).format(a=decoy[0])]
    rng.shuffle(roots)
    state = " ".join(roots + lines)
    question = rng.choice(QUESTIONS).format(t=chain[-1])
    meta = {"chain": chain, "root": root, "claims": claims, "trace_values": vals,
            "decoy": decoy, "decoy_root": d_root, "decoy_claims": d_claims, "decoy_values": d_vals}
    fp = fingerprint(chain, root, claims, decoy, d_root, d_claims)
    return make_item("liars", k, state, question, ["yes", "no"],
                     [1.0, 0.0] if want == 0 else [0.0, 1.0], meta, deterministic=True, fp=fp)


def generate(ks, per_cell, seed):
    return balanced(make, ks, per_cell, seed, n_options=2)


# ---- round-trip parser: recompute the answer from the rendered text only ----

def parse(item):
    roots, says = {}, {}          # says[a] = (b, claim_is_truth)
    for s in re.split(r"(?<=\.) ", item["state"]):
        s = s.strip()
        m = re.fullmatch(r"(\w+) (always tells the truth|is honest|always lies|is a liar)\.", s)
        if m:
            roots[m.group(1)] = m.group(2) in ("always tells the truth", "is honest")
            continue
        m = (re.fullmatch(r"(\w+) says that (\w+) (tells the truth|lies)\.", s)
             or re.fullmatch(r"(\w+) says (\w+) (is honest|is a liar)\.", s)
             or re.fullmatch(r"According to (\w+), (\w+) (tells the truth|lies)\.", s))
        assert m, s
        says[m.group(1)] = (m.group(2), m.group(3) in ("tells the truth", "is honest"))
    target = re.search(r"(?:Does|Is) (\w+) ", item["question"]).group(1)
    return roots, says, target


def verify(item):
    roots, says, target = parse(item)
    # walk back from the target to a root, then evaluate forward (a different route than make())
    path = [target]
    while path[-1] not in roots:
        path.append(says[path[-1]][0])
    val = roots[path[-1]]
    for person in reversed(path[:-1]):
        b, claim_truth = says[person]
        val = (claim_truth == val)
    return (val == (item["answer_index"] == 0)) and len(path) - 1 == item["k"]


def _count(it, words):
    return sum(it["state"].count(w) for w in words)


HEURISTICS = {
    "lie_words": lambda it: _count(it, ["lies", "liar"]),
    "lie_words_parity": lambda it: _count(it, ["lies", "liar"]) % 2,
    "truth_words": lambda it: _count(it, ["truth", "honest"]),
    "roots_honest": lambda it: float(it["meta"]["root"]) + float(it["meta"]["decoy_root"]),
    "target_first_mention": lambda it: it["state"].find(it["meta"]["chain"][-1]) / len(it["state"]),
    "tokens": lambda it: approx_tokens(it["state"] + it["question"]),
}
