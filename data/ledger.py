"""Family `ledger`: is the final balance at least $T?  (sequential state update)

Two accounts; k transactions on each, interleaved in random order. The rule that
makes this sequential rather than a sum: a withdrawal or payment that would take an
account below $0 is declined and changes nothing. Whether a transaction goes through
therefore depends on the running balance, which depends on everything before it.

Every item has at least one decline on the asked-about account, so "add everything
up" (the naive sum N) is wrong by construction on every "yes" item and right on every
"no" item: exactly chance. The threshold T is placed within $15 of the true final
balance F, on the requested side, and above N.
"""
import itertools
import re

from .common import FIRST_NAMES, approx_tokens, balanced, fingerprint, make_item

RULE = ("{A} starts with ${a} and {B} starts with ${b}. Transactions are processed in the "
        "order listed. A withdrawal or payment that would take an account below $0 is declined "
        "and leaves that balance unchanged.")

DEPOSIT = ["Deposit ${amt} into {acc}.", "{acc} receives a deposit of ${amt}.", "${amt} is paid into {acc}."]
WITHDRAW = ["Withdraw ${amt} from {acc}.", "Pay a ${amt} bill from {acc}.",
            "A ${amt} payment is made from {acc}.", "{acc} is charged ${amt}."]
QUESTIONS = ["After all transactions, is the balance of {acc} at least ${t}?",
             "Once every transaction has been processed, does {acc} hold ${t} or more?"]


def simulate(start, txs):
    """txs: list of (sign, amount). Returns (final, trace) where trace has one entry per tx."""
    bal, trace = start, []
    for sign, amt in txs:
        if sign < 0 and bal - amt < 0:
            trace.append({"amount": -amt, "declined": True, "balance": bal})
        else:
            bal += sign * amt
            trace.append({"amount": sign * amt, "declined": False, "balance": bal})
    return bal, trace


def _sample_txs(k, rng):
    """k transactions with at least one decline and at least one successful withdrawal."""
    for _ in range(1000):
        start = rng.randint(20, 140)
        txs = []
        for _ in range(k):
            if rng.random() < 0.5:
                txs.append((+1, rng.randint(5, 60)))
            else:
                txs.append((-1, rng.randint(10, 150)))
        final, trace = simulate(start, txs)
        declines = sum(t["declined"] for t in trace)
        ok_withdraw = any(t["amount"] < 0 and not t["declined"] for t in trace)
        if 1 <= declines <= max(1, k // 2) and (ok_withdraw or k <= 2) and final >= 20:
            return start, txs, final, trace
    raise RuntimeError("could not sample a ledger")


def _render(names, starts, lines_by_acc, t, rng):
    """Interleave the two accounts' lines in random order (each account keeps its own order).

    names[0] is always the asked-about account; which account the rule sentence mentions
    first is random, so mention order carries no signal."""
    lines = []
    slots = [0] * len(lines_by_acc[0]) + [1] * len(lines_by_acc[1])
    rng.shuffle(slots)
    idx = [0, 0]
    for s in slots:
        lines.append(lines_by_acc[s][idx[s]])
        idx[s] += 1
    first, second = (0, 1) if rng.random() < 0.5 else (1, 0)
    rule = RULE.format(A=names[first], a=starts[first], B=names[second], b=starts[second])
    question = rng.choice(QUESTIONS).format(acc=names[0], t=t)
    return rule + " " + " ".join(lines), question


def _lines(name, txs, rng):
    out = []
    for sign, amt in txs:
        tpl = rng.choice(DEPOSIT if sign > 0 else WITHDRAW)
        out.append(tpl.format(amt=amt, acc=name))
    return out


def make(k, want, rng):
    """One deterministic item with answer_index == want (0 = yes, 1 = no)."""
    a, b = rng.sample(FIRST_NAMES, 2)
    names = (a + "'s account", b + "'s account")
    # The threshold is drawn first, independently of the label, and the transactions are
    # re-sampled until the true final balance F lands within $10 of it on the wanted side.
    # So the threshold's own value carries no information about the answer, and an
    # approximate balance is not enough to answer.
    t = rng.randint(36, 120)
    for _ in range(20000):
        start, txs, final, trace = _sample_txs(k, rng)
        naive = start + sum(s * m for s, m in txs)             # what "add everything up" gives
        if want == 0 and t <= final <= t + 9 and naive < t:    # yes: N < T <= F
            break
        if want == 1 and t - 10 <= final <= t - 1:              # no:  N < F < T
            break
    else:
        raise RuntimeError("could not place the final balance near the threshold")
    start2, txs2, _, _ = _sample_txs(k, rng)
    assert final > naive and (final >= t) == (want == 0) and naive < t
    state, question = _render(names, (start, start2),
                              (_lines(names[0], txs, rng), _lines(names[1], txs2, rng)), t, rng)
    meta = {"start": start, "txs": txs, "final": final, "naive_sum": naive, "threshold": t,
            "trace": trace, "asked_account": names[0], "other_account": names[1],
            "other_start": start2, "other_txs": txs2}
    fp = fingerprint(start, txs, t, start2, txs2)
    return make_item("ledger", k, state, question, ["yes", "no"],
                     [1.0, 0.0] if want == 0 else [0.0, 1.0], meta, deterministic=True, fp=fp)


def generate(ks, per_cell, seed):
    return balanced(make, ks, per_cell, seed, n_options=2)


# ---- round-trip parser: recompute the answer from the rendered text only ----

def parse(item):
    s = item["state"]
    m = re.match(r"(.+?) starts with \$(\d+) and (.+?) starts with \$(\d+)\. ", s)
    assert m, s
    starts = {m.group(1): int(m.group(2)), m.group(3): int(m.group(4))}
    rest = s[m.end():]
    rest = rest.split("unchanged. ", 1)[1]
    txs = {n: [] for n in starts}
    for line in re.split(r"(?<=\.) (?=[A-Z$])", rest):
        line = line.strip()
        if not line:
            continue
        pats = [
            (r"Deposit \$(\d+) into (.+)\.", +1), (r"(.+) receives a deposit of \$(\d+)\.", +1),
            (r"\$(\d+) is paid into (.+)\.", +1),
            (r"Withdraw \$(\d+) from (.+)\.", -1), (r"Pay a \$(\d+) bill from (.+)\.", -1),
            (r"A \$(\d+) payment is made from (.+)\.", -1), (r"(.+) is charged \$(\d+)\.", -1),
            (r"A deposit of \$(\d+) into (.+) is expected here with probability ([0-9.]+)\.", "u"),
            (r"With probability ([0-9.]+), \$(\d+) arrives in (.+) at this point\.", "u2"),
        ]
        for pat, sign in pats:
            mm = re.fullmatch(pat, line)
            if not mm:
                continue
            g = mm.groups()
            if sign == "u":
                txs[g[1]].append(("u", int(g[0]), float(g[2])))
            elif sign == "u2":
                txs[g[2]].append(("u", int(g[1]), float(g[0])))
            elif g[0].isdigit():
                txs[g[1]].append((sign, int(g[0])))
            else:
                txs[g[0]].append((sign, int(g[1])))
            break
        else:
            raise AssertionError("unparsed line: " + line)
    q = item["question"]
    acc = re.search(r"balance of (.+?) at least|does (.+?) hold", q)
    acc = acc.group(1) or acc.group(2)
    t = int(re.search(r"\$(\d+)", q).group(1))
    return starts, txs, acc, t


def verify(item):
    starts, txs, acc, t = parse(item)
    seq = txs[acc]
    unc = [i for i, x in enumerate(seq) if x[0] == "u"]
    p_yes = 0.0
    for arrive in itertools.product([True, False], repeat=len(unc)):
        bal, w = starts[acc], 1.0
        for i, x in enumerate(seq):
            if x[0] == "u":
                j = unc.index(i)
                w *= x[2] if arrive[j] else 1 - x[2]
                if arrive[j]:
                    bal += x[1]
            elif x[0] > 0:
                bal += x[1]
            elif bal - x[1] >= 0:
                bal -= x[1]
        if bal >= t:
            p_yes += w
    return abs(p_yes - item["answer_distribution"][0]) < 1e-6


HEURISTICS = {
    "naive_sum_says_yes": lambda it: float(it["meta"]["naive_sum"] >= it["meta"]["threshold"]),
    "threshold": lambda it: it["meta"]["threshold"],
    "start": lambda it: it["meta"]["start"],
    "n_deposits": lambda it: sum(s > 0 for s, _ in it["meta"]["txs"]),
    "sum_deposits": lambda it: sum(m for s, m in it["meta"]["txs"] if s > 0),
    "sum_withdrawals": lambda it: sum(m for s, m in it["meta"]["txs"] if s < 0),
    "max_withdrawal": lambda it: max([m for s, m in it["meta"]["txs"] if s < 0] or [0]),
    "tokens": lambda it: approx_tokens(it["state"] + it["question"]),
}
