"""Family `policy`: under this refund policy, is the case approved?  (rule composition)

Structure: a chain of k rules over a bank of base conditions. Rule i says
"a case is P_i if it is P_{i-1} and B_i[, unless C_i]"; P_k is "approved" and the first
rule starts from a base condition B_0. Deciding the case takes k rule applications.
Decoy rules about other predicates are mixed in and all rules are shuffled, so the
chain has to be found before it can be followed.

Three answers, balanced:
  approved:      every condition on the chain is satisfied by the stated facts.
  not approved:  exactly one condition on the chain fails; its position `fail_pos`
                 (0..k) is stored in meta.
  cannot tell:   one condition on the chain is not stated in the case and everything
                 else holds, so the outcome genuinely depends on the missing fact.

Facts about decoy conditions are stated only some of the time, so "a fact is missing"
is by itself no signal. Labels come from a generic three-valued forward-chaining
evaluator run on the rendered rule set, and are cross-checked by enumerating every
completion of the missing facts.
"""
import itertools
import re

from .common import FIRST_NAMES, approx_tokens, balanced, cap, fingerprint, make_item

# (key, clause when true, clause when false)
CONDITIONS = [
    ("member", "the customer is a member", "the customer is not a member"),
    ("receipt", "the receipt is available", "no receipt is available"),
    ("unopened", "the item is unopened", "the item has been opened"),
    ("damaged", "the item arrived damaged", "the item arrived undamaged"),
    ("final_sale", "the item was sold as final sale", "the item was not sold as final sale"),
    ("online", "the order was placed online", "the order was placed in a store"),
    ("gift", "the item was bought as a gift", "the item was not bought as a gift"),
    ("card", "the order was paid by card", "the order was paid in cash"),
    ("domestic", "the delivery address is domestic", "the delivery address is international"),
    ("over_100", "the item cost more than $100", "the item cost $100 or less"),
    ("within_30", "the order was placed within the last 30 days", "the order was placed more than 30 days ago"),
    ("electronics", "the item is an electronic device", "the item is not an electronic device"),
    ("in_stock", "the item is still in stock", "the item is out of stock"),
    ("id_verified", "the customer's identity has been verified", "the customer's identity has not been verified"),
    ("prior_refund", "the customer has received a refund this year", "the customer has received no refund this year"),
    ("subscription", "the order is part of a subscription", "the order is a one-time order"),
    ("warranty", "the warranty is registered", "the warranty is not registered"),
    ("promo", "the order used a promotional code", "the order did not use a promotional code"),
    ("dispute", "the customer has an open payment dispute", "the customer has no open payment dispute"),
    ("perishable", "the item is perishable", "the item is not perishable"),
    ("customized", "the item was customized", "the item was not customized"),
    ("packaging", "the original packaging is intact", "the original packaging is missing"),
    ("tags", "the tags are still attached", "the tags have been removed"),
    ("bulk", "the order contains more than five units", "the order contains five units or fewer"),
    ("express", "the order used express shipping", "the order used standard shipping"),
    ("delivered", "the delivery has been confirmed", "the delivery has not been confirmed"),
    ("store_credit", "the customer asked for store credit", "the customer asked for a cash refund"),
    ("flagged", "the account has been flagged for review", "the account has not been flagged for review"),
    ("business", "the order was placed by a business account", "the order was placed by a personal account"),
    ("extended", "an extended return window was purchased", "no extended return window was purchased"),
]
COND = {key: (pos, neg) for key, pos, neg in CONDITIONS}
CLAUSE_TO_LIT = {pos: (key, True) for key, pos, neg in CONDITIONS}
CLAUSE_TO_LIT.update({neg: (key, False) for key, pos, neg in CONDITIONS})

PREDICATES = ["eligible", "covered", "qualified", "cleared", "protected", "standard", "priority",
              "endorsed", "certified", "entitled", "authorized", "reviewed", "trusted", "expedited",
              "insured", "screened", "accepted", "preferred", "supported", "exempt", "recognized",
              "validated", "escalated", "sponsored"]
FINAL = "approved"
OPTIONS = ["approved", "not approved", "cannot be determined"]
RULE_TPL = ["A case is {p} if {body}.", "A case counts as {p} when {body}."]
QUESTIONS = ["Under this policy, is {name}'s case approved?",
             "According to the policy, is the refund for {name} approved?"]


def clause(lit):
    key, val = lit
    return COND[key][0] if val else COND[key][1]


def render_rule(rule, rng):
    """rule: dict(head, pre (predicate or None), lits [(key,val)...], unless [(key,val)...])."""
    parts = []
    if rule["pre"]:
        parts.append(f"it is {rule['pre']}")
    parts += [clause(l) for l in rule["lits"]]
    body = " and ".join(parts)
    if rule["unless"]:
        assert len(rule["unless"]) == 1            # one exception per rule; clauses may contain " or "
        body += ", unless " + clause(rule["unless"][0])
    return rng.choice(RULE_TPL).format(p=rule["head"], body=body)


# ---- generic evaluator over the rendered rule set (three-valued) ----

def evaluate(rules, facts):
    """facts: dict key -> bool for stated facts. Returns dict predicate -> True/False/None."""
    def lit_val(l):
        key, val = l
        if key not in facts:
            return None
        return facts[key] == val

    def AND(vals):
        if any(v is False for v in vals):
            return False
        if any(v is None for v in vals):
            return None
        return True

    def OR(vals):
        if any(v is True for v in vals):
            return True
        if any(v is None for v in vals):
            return None
        return False

    def NOT(v):
        return None if v is None else (not v)

    heads = {r["head"] for r in rules}
    val = {h: False for h in heads}           # closed world for predicates: only rules derive them
    for _ in range(len(rules) + 2):           # iterate to a fixpoint (rules may be in any order)
        new = {}
        for h in heads:
            bodies = []
            for r in rules:
                if r["head"] != h:
                    continue
                parts = [lit_val(l) for l in r["lits"]]
                if r["pre"]:
                    parts.append(val[r["pre"]])
                if r["unless"]:
                    parts.append(NOT(OR([lit_val(l) for l in r["unless"]])))
                bodies.append(AND(parts))
            new[h] = OR(bodies)
        if new == val:
            break
        val = new
    return val


def _build(k, rng):
    """Chain rules 1..k plus decoy rules, all over distinct conditions where it matters."""
    keys = [c[0] for c in CONDITIONS]
    rng.shuffle(keys)
    preds = rng.sample(PREDICATES, k - 1 + max(1, k // 4))
    chain_preds = preds[: k - 1] + [FINAL]
    chain = []
    b0 = (keys.pop(), rng.random() < 0.5)
    for i in range(k):
        rule = {"head": chain_preds[i], "pre": chain_preds[i - 1] if i > 0 else None,
                "lits": [(keys.pop(), rng.random() < 0.5)], "unless": []}
        if i == 0:
            rule["lits"].insert(0, b0)
        if rng.random() < 0.35 and len(keys) > (k - i) + len(preds) - (k - 1):   # keep keys for chain + decoys
            rule["unless"] = [(keys.pop(), rng.random() < 0.5)]
        chain.append(rule)
    decoys = []
    for dp in preds[k - 1:]:
        pre = rng.choice(chain_preds[:-1] + [None]) if k > 1 else None
        lits = [(keys.pop(), rng.random() < 0.5)]                # decoy conditions are distinct from chain ones
        decoys.append({"head": dp, "pre": pre, "lits": lits, "unless": []})
    return chain, decoys


def _chain_literals(chain):
    """Ordered list of (position, literal, kind) along the chain; kind 'need' must hold, 'unless' must not."""
    out = []
    for i, r in enumerate(chain):
        for l in r["lits"]:
            out.append((i, l, "need"))
        for l in r["unless"]:
            out.append((i, l, "unless"))
    return out


def make(k, want, rng):
    """One item with answer_index == want (0 approved, 1 not approved, 2 cannot tell)."""
    chain, decoys = _build(k, rng)
    lits = _chain_literals(chain)
    facts = {}
    for _, (key, val), kind in lits:
        facts[key] = val if kind == "need" else (not val)     # satisfy everything
    fail_pos = None
    if want == 1:
        pos, (key, val), kind = rng.choice(lits)
        facts[key] = not facts[key]                            # break exactly one condition
        fail_pos = pos
    elif want == 2:
        pos, (key, val), kind = rng.choice(lits)
        del facts[key]                                         # leave one needed fact unstated
        fail_pos = pos
    # Decoy conditions: the number of unstated conditions is matched across the three answers,
    # so "how many used conditions are missing" carries no signal. Every item leaves m + 1
    # used conditions unstated; for "cannot tell" one of them is the needed chain fact.
    decoy_keys = [key for r in decoys for key, _ in r["lits"]]
    m = rng.randint(0, len(decoy_keys) - 1)
    n_missing_decoys = m if want == 2 else m + 1
    rng.shuffle(decoy_keys)
    for key in decoy_keys[n_missing_decoys:]:
        facts[key] = rng.random() < 0.5
    rules = chain + decoys
    rng.shuffle(rules)
    values = evaluate(rules, facts)
    val = values[FINAL]
    answer = 0 if val is True else (1 if val is False else 2)
    assert answer == want, (k, want, answer)
    name = rng.choice(FIRST_NAMES)
    fact_lines = [cap(clause((key, v))) + "." for key, v in facts.items()]
    rng.shuffle(fact_lines)
    state = ("Policy: " + " ".join(render_rule(r, rng) for r in rules)
             + f" Case of {name}: " + " ".join(fact_lines))
    question = rng.choice(QUESTIONS).format(name=name)
    trace = [{"pred": r["head"], "value": values[r["head"]]} for r in chain]
    meta = {"chain": chain, "decoys": decoys, "facts": facts, "fail_pos": fail_pos,
            "trace": trace, "n_rules": len(rules), "n_facts": len(facts)}
    fp = fingerprint(chain, decoys, sorted(facts.items()))
    dist = [0.0, 0.0, 0.0]
    dist[want] = 1.0
    return make_item("policy", k, state, question, OPTIONS, dist, meta, deterministic=True, fp=fp)


def generate(ks, per_cell, seed):
    return balanced(make, ks, per_cell, seed, n_options=3)


# ---- round-trip parser: recompute the answer from the rendered text only ----

def parse(item):
    s = item["state"]
    policy, case = s.split(" Case of ", 1)
    policy = policy[len("Policy: "):]
    rules = []
    for r in re.split(r"(?<=\.) (?=A case)", policy):
        m = re.fullmatch(r"A case (?:is|counts as) (\w+) (?:if|when) (.+)\.", r.strip())
        assert m, r
        head, body = m.group(1), m.group(2)
        unless = []
        if ", unless " in body:
            body, u = body.split(", unless ")
            unless = [CLAUSE_TO_LIT[u]]
        pre, lits = None, []
        for part in body.split(" and "):
            if part.startswith("it is "):
                pre = part[len("it is "):]
            else:
                lits.append(CLAUSE_TO_LIT[part])
        rules.append({"head": head, "pre": pre, "lits": lits, "unless": unless})
    facts = {}
    for f in re.split(r"(?<=\.) ", case.split(": ", 1)[1]):
        f = f.strip().rstrip(".")
        if f:
            key, val = CLAUSE_TO_LIT[f[0].lower() + f[1:]]
            facts[key] = val
    return rules, facts


def verify(item):
    """Enumerate every completion of the unstated conditions: a different check than Kleene logic."""
    rules, facts = parse(item)
    used = sorted({l[0] for r in rules for l in r["lits"] + r["unless"]})
    missing = [key for key in used if key not in facts]
    outcomes = set()
    for vals in itertools.product([True, False], repeat=len(missing)):
        full = dict(facts, **dict(zip(missing, vals)))
        v = evaluate(rules, full)[FINAL]
        assert v is not None
        outcomes.add(v)
        if len(outcomes) == 2:
            break
    if outcomes == {True}:
        ans = 0
    elif outcomes == {False}:
        ans = 1
    else:
        ans = 2
    return ans == item["answer_index"]


HEURISTICS = {
    "n_facts": lambda it: it["meta"]["n_facts"],
    "n_rules": lambda it: it["meta"]["n_rules"],
    "n_negated_facts": lambda it: sum(1 for v in it["meta"]["facts"].values() if not v),
    "n_unless": lambda it: it["state"].count("unless"),
    "n_missing_used_conditions": lambda it: sum(
        l[0] not in it["meta"]["facts"] for r in it["meta"]["chain"] + it["meta"]["decoys"]
        for l in r["lits"] + r["unless"]),
    "tokens": lambda it: approx_tokens(it["state"] + it["question"]),
}
