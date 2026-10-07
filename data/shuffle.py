"""Family `shuffle` (the object swaps of the paper): after a series of swaps, who holds a given object?  (state tracking)

Structure: five people each hold one of five different objects. Then pairs of people
swap what they hold, one swap after another, and the question asks who holds one named
object at the end. Each swap that involves the object's current holder moves it, so the
answer is the end of a chain of k moves.

As in `liars`, as much material again is mixed in that does not matter for the answer: k
further swaps between people who do not hold the asked object at that moment, at random
positions. So the model must first tell which swaps move the object, then follow them.
Unlike `liars`, the sentences cannot be listed in random order: the order of the swaps is
part of the puzzle.
"""
import re
from collections import Counter

from .common import FIRST_NAMES, approx_tokens, balanced, fingerprint, make_item

OBJECTS = ["ball", "book", "scarf", "cup", "umbrella", "hat", "key", "pen", "box", "coin"]
HOLD = ["{a} has the {o}", "{a} holds the {o}"]
SWAP = ["Then {a} and {b} swap.", "Then {a} swaps with {b}.", "Then {a} and {b} trade."]
QUESTIONS = ["Who has the {o} at the end?", "At the end, who holds the {o}?"]
N = 5


def make(k, want, rng):
    """One item with answer_index == want (the option position of the final holder)."""
    while True:
        start = rng.randrange(N)
        holder, kinds = start, [True] * k + [False] * k
        rng.shuffle(kinds)
        swaps, trace = [], [start]
        for relevant in kinds:
            if relevant:
                a, b = holder, rng.choice([p for p in range(N) if p != holder])
                holder = b
            else:
                a, b = rng.sample([p for p in range(N) if p != holder], 2)
            swaps.append((a, b) if rng.random() < 0.5 else (b, a))
            trace.append(holder)
        if holder == want:
            break
    people = rng.sample(FIRST_NAMES, N)
    things = rng.sample(OBJECTS, N)
    hold = rng.choice(HOLD)
    parts = [hold.format(a=p, o=t) for p, t in zip(people, things)]
    state = ", ".join(parts[:-1]) + ", and " + parts[-1] + ". "
    state += " ".join(rng.choice(SWAP).format(a=people[a], b=people[b]) for a, b in swaps)
    question = rng.choice(QUESTIONS).format(o=things[start])
    meta = {"people": people, "objects": things, "target": things[start], "start": start, "swaps": swaps,
            "relevant": kinds, "trace_holders": trace}
    fp = fingerprint(people, things, start, [sorted(s) for s in swaps])
    dist = [1.0 if i == want else 0.0 for i in range(N)]
    return make_item("shuffle", k, state, question, people, dist, meta, deterministic=True, fp=fp)


def generate(ks, per_cell, seed):
    return balanced(make, ks, per_cell, seed, n_options=N)


# ---- round-trip parser: recompute the answer from the rendered text only ----

def parse(item):
    first, *rest = re.split(r"(?<=\.) ", item["state"])
    holds = re.findall(r"(\w+) (?:has|holds) the (\w+)", first)
    swaps = []
    for s in rest:
        m = (re.fullmatch(r"Then (\w+) and (\w+) (?:swap|trade)\.", s.strip())
             or re.fullmatch(r"Then (\w+) swaps with (\w+)\.", s.strip()))
        assert m, s
        swaps.append((m.group(1), m.group(2)))
    target = re.search(r"(?:has|holds) the (\w+)", item["question"]).group(1)
    return holds, swaps, target


def verify(item):
    holds, swaps, target = parse(item)
    # follow all five objects by name (make() follows one holder by position), and count the swaps that moved the
    # asked object
    owner = {p: o for p, o in holds}
    moved = 0
    for a, b in swaps:
        moved += target in (owner[a], owner[b])
        owner[a], owner[b] = owner[b], owner[a]
    final = [p for p, o in owner.items() if o == target]
    return (len(owner) == N and final == [item["options"][item["answer_index"]]] and moved == item["k"]
            and len(swaps) == 2 * item["k"])


def _mentions(it):
    return Counter(p for s in it["meta"]["swaps"] for p in s)


HEURISTICS = {
    "last_swap_first_person": lambda it: it["meta"]["swaps"][-1][0],
    "last_swap_second_person": lambda it: it["meta"]["swaps"][-1][1],
    "most_mentioned_person": lambda it: max(range(N), key=lambda p: (_mentions(it)[p], -p)),
    "first_holder": lambda it: it["meta"]["start"],
    "first_swap_first_person": lambda it: it["meta"]["swaps"][0][0],
    "tokens": lambda it: approx_tokens(it["state"] + it["question"]),
}
