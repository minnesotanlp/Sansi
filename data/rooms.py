"""Family `rooms`: can you walk from the start room to the target room?  (search)

Structure: two components of equal size. Component A contains the start room and a
spine of exactly k doors; extra rooms hang off the spine and a few extra doors add
cycles without changing any distance. Component B is a decoy of the same size.

  positive: the target is a room of A at distance exactly k from the start.
  negative: the target is a room of B, chosen to have the same number of doors as a
            room of A at distance k, so the target's degree cannot predict the label.

To answer "no" the model must exhaust A, which is k levels deep, so both labels cost
about k steps. Doors are listed in random order with random orientation.
"""
import itertools
import re
from collections import deque

from .common import approx_tokens, balanced, fingerprint, make_item

CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"
ROOM_NAMES = [a + b for a in CONSONANTS for b in CONSONANTS if a != b]  # e.g. "KX": no real words

QUESTIONS = [
    "Starting in room {s} and walking only through open doors, can you reach room {t}?",
    "You are in room {s}. Using only open doors, is it possible to get to room {t}?",
    "Is there a route from room {s} to room {t} that uses only open doors?",
]


def bfs(n, edges, src, open_mask=None):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(edges):
        if open_mask is None or open_mask[i]:
            adj[a].append(b)
            adj[b].append(a)
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _grow_tree(nodes, placed, rng):
    """Attach `nodes` one at a time to already placed nodes: a tree, no shortcuts."""
    placed, edges = list(placed), []
    for v in nodes:
        edges.append((rng.choice(placed), v))
        placed.append(v)
    return edges


def _add_level_edges(edges, n, src, count, rng):
    """Extra doors between rooms whose BFS levels differ by at most one.

    They create cycles but leave every distance from `src` unchanged."""
    lv = bfs(n, edges, src)
    have = {frozenset(e) for e in edges}
    cand = [(a, b) for a in lv for b in lv
            if a < b and abs(lv[a] - lv[b]) <= 1 and frozenset((a, b)) not in have]
    return edges + rng.sample(cand, min(count, len(cand)))


def _structure(k, rng, branch=None):
    """Two components of `size` rooms each. Returns (edges, n, spine, b_nodes)."""
    branch = k if branch is None else branch
    size = k + 1 + branch
    n = 2 * size
    a_nodes = list(range(size))
    b_nodes = list(range(size, n))
    spine = a_nodes[: k + 1]                                   # start = 0 ... spine[k]
    e_a = list(zip(spine, spine[1:])) + _grow_tree(a_nodes[k + 1:], spine, rng)
    e_a = _add_level_edges(e_a, n, 0, count=max(1, k // 3), rng=rng)
    e_b = _grow_tree(b_nodes[1:], b_nodes[:1], rng)
    e_b = _add_level_edges(e_b, n, b_nodes[0], count=max(1, k // 3), rng=rng)
    return e_a + e_b, n, spine, b_nodes


def _render(edges, n, target, p_open, rng, family, k, dist_yes, meta, deterministic):
    names = rng.sample(ROOM_NAMES, n)
    order = list(range(len(edges)))
    rng.shuffle(order)
    doors = []
    for i in order:
        a, b = edges[i]
        if rng.random() < 0.5:
            a, b = b, a
        s = f"{names[a]}-{names[b]}"
        if i in p_open:
            s += f" (opens with probability {p_open[i]:.1f})"
        doors.append(s)
    state = "Doors between rooms (two-way): " + "; ".join(doors) + "."
    question = rng.choice(QUESTIONS).format(s=names[0], t=names[target])
    meta = dict(meta, names=names)
    fp = fingerprint(sorted(tuple(sorted(e)) for e in edges), target, sorted(p_open.items()))
    return make_item(family, k, state, question, ["yes", "no"], [dist_yes, 1 - dist_yes],
                     meta, deterministic=deterministic, fp=fp)


def make(k, want, rng):
    """One deterministic item with answer_index == want (0 = yes, 1 = no)."""
    for _ in range(100):
        edges, n, spine, b_nodes = _structure(k, rng)
        dist = bfs(n, edges, 0)
        deg = [0] * n
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        ref = rng.choice([v for v, d in dist.items() if d == k])
        same_deg = [v for v in b_nodes if deg[v] == deg[ref]]
        if same_deg:
            break
    else:
        raise RuntimeError("no decoy room with a matching number of doors")
    target = ref if want == 0 else rng.choice(same_deg)
    reachable = target in dist
    assert reachable == (want == 0)
    assert not reachable or dist[target] == k
    levels = [[v for v, d in dist.items() if d == lv] for lv in range(max(dist.values()) + 1)]
    meta = {"edges": edges, "start": 0, "target": target, "trace_levels": levels,
            "target_degree": deg[target], "n_rooms": n}
    return _render(edges, n, target, {}, rng, "rooms", k, 1.0 if reachable else 0.0, meta, True)


def generate(ks, per_cell, seed):
    return balanced(make, ks, per_cell, seed, n_options=2)


# ---- round-trip parser: recompute the answer from the rendered text only ----

def parse(item):
    body = item["state"].split(": ", 1)[1].rstrip(".")
    edges, p_open = [], {}
    for i, door in enumerate(body.split("; ")):
        m = re.match(r"(\w+)-(\w+)(?: \(opens with probability ([0-9.]+)\))?$", door)
        assert m, door
        edges.append((m.group(1), m.group(2)))
        if m.group(3):
            p_open[i] = float(m.group(3))
    q = item["question"]
    start = re.search(r"room (\w+)", q).group(1)
    target = re.findall(r"room (\w+)", q)[-1]
    return edges, p_open, start, target


def verify(item):
    """Recompute P(yes) from the text with a different algorithm (union-find)."""
    edges, p_open, start, target = parse(item)
    coin = sorted(p_open)

    def reach(mask):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, (a, b) in enumerate(edges):
            if mask[i]:
                parent[find(a)] = find(b)
        return find(start) == find(target)

    p_yes = 0.0
    for states in itertools.product([True, False], repeat=len(coin)):
        mask = [True] * len(edges)
        w = 1.0
        for i, s in zip(coin, states):
            mask[i] = s
            w *= p_open[i] if s else 1 - p_open[i]
        if reach(mask):
            p_yes += w
    return abs(p_yes - item["answer_distribution"][0]) < 1e-6


# ---- cheap heuristics a shortcut learner might use; the probe checks they are at chance ----

HEURISTICS = {
    "target_degree": lambda it: it["meta"]["target_degree"],
    "n_doors": lambda it: len(it["meta"]["edges"]),
    "target_first_mention": lambda it: it["state"].find(it["meta"]["names"][it["meta"]["target"]]) / len(it["state"]),
    "tokens": lambda it: approx_tokens(it["state"] + it["question"]),
}
