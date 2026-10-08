"""Build the main set: the training, development and test items of the decision suite.

  python -m data.build_main --raw <raw data folder> --tokenizer <Ouro-1.4B> --out datasets/main

Six item types (meta.layer) x test groups (meta.tier: in_dist / near / far; training items have tier "train"):
  A  classification     decision-v7 classification sources; far: Emotion, TweetEval, Yahoo Answers, Kev v9 dev
  B  multi-step         ProofWriter, CLUTRR, MuSiQue, Kev rules and policies; near: longer chains, NatLang, 4 hops;
                        far: FOLIO, BBH, HoVer, Kev v9 dev held-out rules and policies
  C  uncertain          C1 unanswerable: MuSiQue minimal pairs, SQuAD 2.0; far: Kev v9 dev unknowable pairs.
                        C2 soft or exact probabilities: ChaosNLI; near: ChaosNLI-MNLI; far: Sys1Cal-v1
  D  long documents     MuSiQue / HotpotQA with the datasets' own distractor paragraphs, the in-distribution test at
                        three lengths per question; far: ContractNLI, QuALITY, Kev v9 dev buried states
  E  sentence pairs     MNLI, BoolQ (decision-v7); near: MNLI mismatched; far: ANLI dev, QNLI, PAWS, WANLI, Kev v9 dev
  F  knowledge          test only: SciQ, ARC, MMLU, MMLU-Pro, CommonsenseQA, Kev v9 dev
  JB JevBench           public easy / standard / hard, test only, never split (meta.tier "jevbench")

Layout: train/<source>.jsonl.gz; every test source is split into dev/<source> (20%) and test/<source> (80%) by
component (see assign_dev: items sharing a group, a state, a story, a sub-question or most of their sentences stay
together); JevBench goes to test/ whole. Training is drawn first; a test candidate that repeats training material (same
state, or >= 0.8 of its sentences) is never drawn. manifest.json has the counts, report.txt the checks done while
building.

Unanswerable items (meta.unanswerable) have no correct option: answer_index is None and the target is uniform. They
count as correct when not hard-answered (max probability < (1 + 1/K) / 2). No model is called while building; the only
text we write is question stems and option definitions for label sets without a Kev wording (TWEET, YAHOO, CNLI).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re

import numpy as np

from . import real
from .check_overlap import sentence_set
from .common import fingerprint, write_jsonl

SEED = 20260930
DEV_SHARE = 0.2
REPORT = []


def note(*parts):
    line = " ".join(str(p) for p in parts)
    REPORT.append(line)
    print(line, flush=True)


# ---------------------------------------------------------------- helpers

def rng_for(*parts):
    return random.Random("%d/%s" % (SEED, "/".join(map(str, parts))))


def unique(items):
    """First item of each distinct text (fingerprint: dataset, state, question, options)."""
    seen, out = set(), []
    for it in items:
        if it["fingerprint"] not in seen:
            seen.add(it["fingerprint"])
            out.append(it)
    return out


def sample(items, n, *key):
    items = list(items)
    rng_for("sample", *key).shuffle(items)
    return items[:n]


def stratified(items, keys, total, *tag):
    """total split as evenly as possible over the groups of keys[0] (a group smaller than its share gives everything
    it has, the rest goes to the other groups), then likewise inside each group with keys[1:]; random at the end."""
    if not keys:
        return sample(items, total, *tag)
    groups = collections.defaultdict(list)
    for it in items:
        groups[keys[0](it)].append(it)
    rest, remaining, alloc = sorted(groups, key=str), total, {}
    while rest:
        share = remaining / len(rest)
        small = [g for g in rest if len(groups[g]) <= share]
        if not small:
            break
        for g in small:
            alloc[g] = len(groups[g])
            remaining -= len(groups[g])
            rest.remove(g)
    base, extra = divmod(remaining, len(rest)) if rest else (0, 0)
    for i, g in enumerate(rest):
        alloc[g] = base + (1 if i < extra else 0)
    out = []
    for g in sorted(groups, key=str):
        out += stratified(groups[g], keys[1:], alloc[g], *tag, g)
    return out


K = lambda it: it["k"]
ANS = lambda it: it["answer_index"]


def tag(items, layer, tier, source, group=None):
    for it in items:
        m = it["meta"]
        m["layer"], m["tier"], m["source_set"] = layer, tier, source
        m["group"] = str(group(it)) if group else it["fingerprint"]
    return items


def uniform(it, **extra):
    """Unanswerable: uniform target, no correct option."""
    n = len(it["options"])
    it["answer_distribution"] = [round(1.0 / n, 6)] * n
    it["answer_index"] = None
    it["deterministic"] = False
    it["meta"]["unanswerable"] = True
    it["meta"].update(extra)
    return it


def letter_options(texts):
    return ["%s: %s" % (chr(ord("a") + i), t.strip()) for i, t in enumerate(texts)]


def jstate(d):
    return json.dumps(d, indent=2, ensure_ascii=False)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def with_key(it, text):
    """The source text that identifies this example (used to drop examples Kev v9 dev already has)."""
    it["meta"]["text_key"] = norm(text)
    return it


# ---------------------------------------------------------------- Kev

def kev_group(it):
    """Kev's group: one case with its variants (compositional: relevant / irrelevant field x a / b; policies: the a / b
    contrastive pair; other sources: the record alone)."""
    return it["meta"].get("group_id") or "record-%d" % it["meta"]["record"]


def kev_groups(raw):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in real.kev(raw, "decision_v7", "train"):
        by[it["meta"]["source"]][kev_group(it)].append(it)
    return by


def kev_take(groups, n, skip, ok, *key):
    """Whole Kev groups in a fixed random order, leaving out the groups in skip, until n questions. Kev repeats some
    questions word for word inside a group (decision-v7 compositional: 1,680 questions, 1,260 distinct); a repeat is
    left out, and so is a question failing ok(). Returns the questions and the groups used."""
    ids = sorted(groups)
    rng_for("kev", *key).shuffle(ids)
    seen, out, used = set(), [], []
    for g in ids:
        if len(out) >= n:
            break
        if g in skip:
            continue
        new = [it for it in unique(groups[g]) if it["fingerprint"] not in seen and ok(it)]
        seen |= {it["fingerprint"] for it in groups[g]}
        if new:
            out += new
            used.append(g)
    return out[:n], used


def kev_v9(raw):
    by = collections.defaultdict(list)
    for it in real.kev(raw, "transfer_v9", "development"):
        if it["meta"]["variant"] == "clean":
            by[it["meta"]["source"]].append(it)
    return by


def kev_unknowable_pairs(v9):
    """Kev v9 dev unknowable records with their controls. Some unknowable states are shared word for word by two
    records (the removed evidence was what told them apart, so their Kev labels differ); such pairs are merged into one
    group holding each distinct text once. pair_id is the group's first control id."""
    ctrl = {it["meta"]["id"]: it for it in v9["unknowable_control"]}
    pairs = [(uniform(u, kev_label=u["answer_index"]), ctrl[u["meta"]["control_id"]]) for u in v9["unknowable"]]
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    first = {}
    for i, (u, c) in enumerate(pairs):
        for fp in (u["fingerprint"], c["fingerprint"]):
            if fp in first:
                parent[find(i)] = find(first[fp])
            first.setdefault(fp, i)
    comps = collections.defaultdict(list)
    for i in range(len(pairs)):
        comps[find(i)].append(i)
    out = []
    for members in comps.values():
        pid = min(pairs[i][1]["meta"]["id"] for i in members)
        for it in unique([pairs[i][0] for i in members] + [pairs[i][1] for i in members]):
            it["meta"]["pair_id"] = pid
            it["meta"]["pair_member"] = "unanswerable" if it["meta"].get("unanswerable") else "answerable"
            out.append(it)
    note("Kev v9 dev unknowable pairs: %d pairs -> %d groups, %d unknowable and %d control texts" % (
        len(pairs), len(comps), sum(bool(it["meta"].get("unanswerable")) for it in out),
        sum(not it["meta"].get("unanswerable") for it in out)))
    return out


# ---------------------------------------------------------------- MuSiQue (B, C1, D)

def _par(p):
    return "%s: %s" % (p["title"], p["paragraph_text"])


def _mq_options(r, pool, rng, n=4):
    """Four options: the final answer, the question's intermediate answers (what a model that stops early would say),
    then answers of other questions; in random order."""
    final = r["answer"].strip()
    seen, opts, kinds = {final.lower()}, [final], ["final"]
    for step, dec in enumerate(r["question_decomposition"][:-1]):
        a = str(dec.get("answer", "")).strip()
        if a and a.lower() not in seen and len(opts) < n:
            opts.append(a)
            kinds.append("intermediate_hop%d" % (step + 1))
            seen.add(a.lower())
    while len(opts) < n:
        a = rng.choice(pool).strip()
        if a and a.lower() not in seen:
            opts.append(a)
            kinds.append("other_question")
            seen.add(a.lower())
    order = list(range(n))
    rng.shuffle(order)
    return [opts[i] for i in order], [kinds[i] for i in order]


def musique_item(r, pool, dataset, split, paragraphs, **meta):
    opts, kinds = _mq_options(r, pool, rng_for("mq_opts", r["id"]))
    q = r["question"].strip() + " (Answer from the paragraphs above.)"
    return real.item(dataset, split, real._hops(r["id"]), "hops", "\n\n".join(_par(p) for p in paragraphs), q, opts,
                     kinds.index("final"), meta={"id": r["id"], "option_kinds": kinds, "n_paragraphs": len(paragraphs),
                                                 "subq": sorted(subq_ids(r)), **meta})


def musique_short(r, pool, dataset, split):
    sup = [p for p in r["paragraphs"] if p.get("is_supporting")]
    rng_for("mq_short", r["id"]).shuffle(sup)
    return musique_item(r, pool, dataset, split, sup)


def musique_lengths(r, pool, dataset, split, versions):
    """The same question and options at several lengths: supporting paragraphs plus 0 / 8 / all of the question's own
    distractor paragraphs."""
    sup = [p for p in r["paragraphs"] if p.get("is_supporting")]
    other = [p for p in r["paragraphs"] if not p.get("is_supporting")]
    rng_for("mq_len", r["id"]).shuffle(other)
    out = []
    for v in versions:
        ps = sup + other[:{"short": 0, "medium": 8, "long": len(other)}[v]]
        rng_for("mq_len_order", r["id"], v).shuffle(ps)
        out.append(musique_item(r, pool, dataset, split, ps, length_version=v))
    return out


def _words(p):
    return set(re.findall(r"[a-z0-9]+", (p["title"] + " " + p["paragraph_text"]).lower()))


def musique_pair(ans, un, pool, dataset, split, n_distract=3):
    """Minimal pair from MuSiQue's official answerable / unanswerable versions of one question: the supporting
    paragraphs plus n_distract distractors present in both versions; in the unanswerable member each supporting
    paragraph the official unanswerable version removed is replaced by the paragraph it put in that shares the most
    words with it (Jaccard), so the two members differ by the missing evidence rather than by an off-topic paragraph.
    Same question and options in both members."""
    key = lambda p: (p["title"], p["paragraph_text"])
    un_keys, ans_keys = {key(p) for p in un["paragraphs"]}, {key(p) for p in ans["paragraphs"]}
    sup = [p for p in ans["paragraphs"] if p.get("is_supporting")]
    removed = [p for p in sup if key(p) not in un_keys]
    added = [p for p in un["paragraphs"] if key(p) not in ans_keys]
    if not removed or len(added) < len(removed):
        return None
    rng = rng_for("mq_pair", ans["id"])
    shared = [p for p in ans["paragraphs"] if not p.get("is_supporting") and key(p) in un_keys]
    rng.shuffle(shared)
    ps = sup + shared[:n_distract]
    rng.shuffle(ps)
    swap, free = {}, list(added)
    for p in removed:
        w = _words(p)
        best = max(free, key=lambda q: (len(w & _words(q)) / max(1, len(w | _words(q))), q["title"]))
        free.remove(best)
        swap[key(p)] = best
    a = musique_item(ans, pool, dataset, split, ps, pair_id=ans["id"], pair_member="answerable")
    u = musique_item(ans, pool, dataset, split, [swap.get(key(p), p) for p in ps], pair_id=ans["id"],
                     pair_member="unanswerable", n_removed=len(removed),
                     removed_titles=[p["title"] for p in removed],
                     replacement_titles=[swap[key(p)]["title"] for p in removed])
    return [a, uniform(u)]


def pairs_by_hop(rows, pairs, pool, split, n, *key):
    """n minimal pairs split evenly over hop counts, from the candidate rows whose pair could be built."""
    built = [p for p in (musique_pair(*pairs[r["id"]], pool, "musique_pair", split) for r in rows) if p]
    return stratified(built, [lambda p: p[0]["k"]], n, *key)


def subq_ids(r):
    return {d["id"] for d in r["question_decomposition"]}


# ---------------------------------------------------------------- HotpotQA (D)

def hotpot_rows(raw, split):
    """Questions whose answer is yes / no, or a comparison whose answer is one of its two supporting titles."""
    part = "train" if split == "train" else "validation"
    df = real._parquets(os.path.join(raw, "hotpotqa__hotpot_qa/distractor/%s-*.parquet" % part))
    out = []
    for r in df.to_dict("records"):
        gold_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        ans = r["answer"].strip()
        if ans.lower() in ("yes", "no"):
            options, gold, qtype = ["no", "yes"], ["no", "yes"].index(ans.lower()), "noul"
        elif r["type"] == "comparison" and len(gold_titles) == 2 and ans.lower() in [t.lower() for t in gold_titles]:
            options, qtype = list(gold_titles), "choice"
            gold = [t.lower() for t in options].index(ans.lower())
        else:
            continue
        paras = [{"title": t, "text": "".join(s).strip(), "gold": t in gold_titles}
                 for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
        if sum(p["gold"] for p in paras) != len(gold_titles):
            continue
        out.append({"id": r["id"], "question": r["question"], "options": options, "gold": gold, "qtype": qtype,
                    "paras": paras, "type": r["type"], "level": r["level"], "titles": gold_titles})
    return out


def hotpot_lengths(r, split, versions):
    """The same question at several lengths: the 2 supporting paragraphs plus 0 / 4 / 8 of its own distractors."""
    gold = [p for p in r["paras"] if p["gold"]]
    other = [p for p in r["paras"] if not p["gold"]]
    rng_for("hp_len", r["id"]).shuffle(other)
    opts, gold_i = list(r["options"]), r["gold"]
    if r["qtype"] == "choice":
        order = [0, 1]
        rng_for("hp_opts", r["id"]).shuffle(order)
        opts, gold_i = [r["options"][i] for i in order], order.index(r["gold"])
    out = []
    for v in versions:
        ps = gold + other[:{"short": 0, "medium": 4, "long": len(other)}[v]]
        rng_for("hp_order", r["id"], v).shuffle(ps)
        out.append(real.item("hotpotqa", split, None, None, "\n\n".join("%s: %s" % (p["title"], p["text"]) for p in ps),
                             r["question"].strip() + " (Answer from the paragraphs above.)", opts, gold_i,
                             qtype=r["qtype"], meta={"id": r["id"], "type": r["type"], "level": r["level"],
                                                     "length_version": v, "n_paragraphs": len(ps)}))
    return out


# ---------------------------------------------------------------- SQuAD 2.0 (C1)

def _span_type(s):
    return "number" if re.search(r"\d", s) else "name" if s[:1].isupper() else "other"


def squad(raw, split):
    """4-option choice. Options: the answer (for an unanswerable question, its annotated plausible answer) and three
    answer spans of other questions on the same passage, of the same kind where possible (number / capitalised /
    other), no option containing another. Unanswerable questions get a uniform target. Returns {False: answerable, True: unanswerable}."""
    name = "train-v2.0.json" if split == "train" else "dev-v2.0.json"
    data = json.load(open(os.path.join(raw, "rajpurkar__squad_v2_official", name)))["data"]
    out = {True: [], False: []}
    for art in data:
        for para in art["paragraphs"]:
            spans = []
            for qa in para["qas"]:
                for a in (qa["answers"] or qa.get("plausible_answers") or [])[:1]:
                    t = a["text"].strip()
                    if t and t.lower() not in [s.lower() for s in spans]:
                        spans.append(t)
            for qa in para["qas"]:
                imp = qa["is_impossible"]
                src = qa.get("plausible_answers") if imp else qa["answers"]
                if not src or not src[0]["text"].strip():
                    continue
                target = src[0]["text"].strip()
                golds = {a["text"].strip().lower() for a in qa["answers"] or []} | {target.lower()}
                cand = [s for s in spans if s.lower() not in golds and target.lower() not in s.lower()
                        and s.lower() not in target.lower()]
                if len(cand) < 3:
                    continue
                rng = rng_for("squad", qa["id"])
                same = [s for s in cand if _span_type(s) == _span_type(target)]
                diff = [s for s in cand if _span_type(s) != _span_type(target)]
                rng.shuffle(same)
                rng.shuffle(diff)
                opts = [target]
                for c in same + diff:
                    if len(opts) < 4 and not any(c.lower() in o.lower() or o.lower() in c.lower() for o in opts):
                        opts.append(c)
                if len(opts) < 4:
                    continue
                rng.shuffle(opts)
                it = real.item("squad2", split, None, None, para["context"],
                               qa["question"].strip() + " (Answer from the passage above.)", opts, opts.index(target),
                               meta={"id": qa["id"], "title": art["title"]})
                out[imp].append(uniform(it, plausible_index=opts.index(target)) if imp else it)
    return out


# ---------------------------------------------------------------- test-only sources

def emotion(raw):
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]   # Kev's question and label order
    df = real._parquets(os.path.join(raw, "dair-ai__emotion/split/test-*.parquet"))
    return [with_key(real.item("emotion", "test", None, None, r["text"], "Which emotion does the writer express?",
                               labels, int(r["label"]), meta={"idx": i}), r["text"])
            for i, r in enumerate(df.to_dict("records"))]


TWEET = {   # question, options in the dataset's label order, qtype; "offensive" is Kev's wording
    "offensive": ("Is this post offensive?", ["no: Not offensive",
                  "yes: Contains insults, threats, profanity directed at someone, or hateful content"], "noul"),
    "hate": ("Does this post express hate toward a group of people, such as immigrants or women?",
             ["no: No hateful content toward a group", "yes: Expresses hatred toward a group, such as immigrants or women"],
             "noul"),
    "irony": ("Is this post ironic?", ["no: Meant literally",
              "yes: Says the opposite of what it literally means, or is sarcastic"], "noul"),
    "sentiment": ("What is the sentiment of this post?", ["negative", "neutral", "positive"], "choice"),
}


def tweet_eval(raw):
    out = []
    for cfg, (q, opts, qtype) in TWEET.items():
        df = real._parquets(os.path.join(raw, "cardiffnlp__tweet_eval/%s/test-*.parquet" % cfg))
        out += [with_key(real.item("tweet_eval_" + cfg, "test", None, None, r["text"], q, opts, int(r["label"]),
                                   qtype=qtype, meta={"idx": i, "task": cfg}), r["text"])
                for i, r in enumerate(df.to_dict("records"))]
    return out


YAHOO = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet",
         "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]


def yahoo(raw):
    df = real._parquets(os.path.join(raw, "community-datasets__yahoo_answers_topics/yahoo_answers_topics/test-*.parquet"))
    out = []
    for r in df.to_dict("records"):
        ans = r["best_answer"].strip()
        ans = ans if len(ans) <= 1500 else ans[:1500].rsplit(" ", 1)[0] + " ..."
        state = "Question title: %s\nQuestion: %s\nBest answer: %s" % (r["question_title"].strip(),
                                                                       r["question_content"].strip(), ans)
        out.append(real.item("yahoo_answers", "test", None, None, state, "Which topic does this question belong to?",
                             YAHOO, int(r["topic"]), meta={"id": str(r["id"])}))
    return out


def qnli(raw):
    df = real._parquets(os.path.join(raw, "nyu-mll__glue/qnli/validation-*.parquet"))
    return [with_key(real.item("qnli", "test", None, None, r["sentence"],
                               'Does the sentence contain the answer to this question: "%s"' % r["question"].strip(),
                               ["no", "yes"], 1 if int(r["label"]) == 0 else 0, qtype="noul",
                               meta={"idx": int(r["idx"])}), r["sentence"]) for r in df.to_dict("records")]


def paws(raw):
    df = real._parquets(os.path.join(raw, "google-research-datasets__paws/labeled_final/test-*.parquet"))
    opts = ["no: Different meaning, even if most words match", "yes: Same meaning, possibly reworded"]
    return [with_key(real.item("paws", "test", None, None, r["sentence1"],
                               'Does this sentence mean the same thing: "%s"' % r["sentence2"].strip(), opts,
                               int(r["label"]), qtype="noul", meta={"id": str(r["id"])}), r["sentence1"])
            for r in df.to_dict("records") if r["sentence1"].strip() != r["sentence2"].strip()]


def wanli(raw):
    rows = [json.loads(l) for l in open(os.path.join(raw, "alisawuffles__WANLI/test.jsonl"))]
    return [real._nli_item("wanli", "test", r["premise"], r["hypothesis"], real.NLI_OPTIONS.index(r["gold"]),
                           meta={"id": str(r["id"])}) for r in rows]


def anli_dev(raw):
    out = []
    for rnd in ("r1", "r2", "r3"):
        for r in real._parquets(os.path.join(raw, "facebook__anli/plain_text/dev_%s-*.parquet" % rnd)).to_dict("records"):
            it = real._nli_item("anli", "test", r["premise"], r["hypothesis"], int(r["label"]),
                                meta={"uid": r["uid"], "round": rnd})
            it["k"], it["depth_kind"] = int(rnd[1]), "anli_round"
            out.append(it)
    return out


def mnli_mismatched(raw):
    rows = real._parquets(os.path.join(raw, "nyu-mll__multi_nli/data/validation_mismatched-*.parquet")).to_dict("records")
    return [real._nli_item("mnli_mismatched", "test", r["premise"], r["hypothesis"], int(r["label"]),
                           meta={"pairID": r["pairID"], "genre": r["genre"]}) for r in rows if int(r["label"]) in (0, 1, 2)]


MMLU_GROUPS = {
    "stem": "abstract_algebra anatomy astronomy college_biology college_chemistry college_computer_science "
            "college_mathematics college_physics computer_security conceptual_physics electrical_engineering "
            "elementary_mathematics high_school_biology high_school_chemistry high_school_computer_science "
            "high_school_mathematics high_school_physics high_school_statistics machine_learning",
    "humanities": "formal_logic high_school_european_history high_school_us_history high_school_world_history "
                  "international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy "
                  "prehistory professional_law world_religions",
    "social_sciences": "econometrics high_school_geography high_school_government_and_politics "
                       "high_school_macroeconomics high_school_microeconomics high_school_psychology human_sexuality "
                       "professional_psychology public_relations security_studies sociology us_foreign_policy",
    "other": "business_ethics clinical_knowledge college_medicine global_facts human_aging management marketing "
             "medical_genetics miscellaneous nutrition professional_accounting professional_medicine virology",
}
MMLU_GROUP_OF = {s: g for g, ss in MMLU_GROUPS.items() for s in ss.split()}
ANSWER_Q = "Which option correctly answers the question?"   # Kev's wording for MMLU and MMLU-Pro


def mmlu(raw):
    df = real._parquets(os.path.join(raw, "cais__mmlu/all/test-*.parquet"))
    return [with_key(real.item("mmlu", "test", None, None,
                               jstate({"subject": r["subject"].replace("_", " "), "question": r["question"].strip()}),
                               ANSWER_Q, letter_options(list(r["choices"])), int(r["answer"]),
                               meta={"idx": i, "subject": r["subject"], "mmlu_group": MMLU_GROUP_OF[r["subject"]]}),
                     r["question"]) for i, r in enumerate(df.to_dict("records"))]


def mmlu_pro(raw):
    df = real._parquets(os.path.join(raw, "TIGER-Lab__MMLU-Pro/data/test-*.parquet"))
    out = []
    for r in df.to_dict("records"):
        opts = [str(o).strip() for o in r["options"]]
        if len(set(opts)) != len(opts):
            continue
        out.append(with_key(real.item("mmlu_pro", "test", None, None,
                                      jstate({"category": r["category"], "question": r["question"].strip()}),
                                      ANSWER_Q, letter_options(opts), int(r["answer_index"]),
                                      meta={"question_id": int(r["question_id"]), "category": r["category"]}),
                            r["question"]))
    return out


def arc(raw, config):
    df = real._parquets(os.path.join(raw, "allenai__ai2_arc/%s/test-*.parquet" % config))
    out = []
    for r in df.to_dict("records"):
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] in labels:
            out.append(real.item(config.lower().replace("-", "_"), "test", None, None,
                                 jstate({"question": r["question"].strip()}), ANSWER_Q, letter_options(texts),
                                 labels.index(r["answerKey"]), meta={"id": r["id"]}))
    return out


def sciq(raw):
    """As Kev's sciq: state {passage, question}, options a-d in a fixed random order."""
    df = real._parquets(os.path.join(raw, "allenai__sciq/data/test-*.parquet"))
    out = []
    for i, r in enumerate(df.to_dict("records")):
        opts = [str(r[k]).strip() for k in ("correct_answer", "distractor1", "distractor2", "distractor3")]
        if len({o.lower() for o in opts}) != 4:
            continue
        order = list(range(4))
        rng_for("sciq", i).shuffle(order)
        out.append(with_key(real.item("sciq", "test", None, None,
                                      jstate({"passage": str(r["support"]).strip(), "question": r["question"].strip()}),
                                      "Which option answers the science question?", letter_options([opts[j] for j in order]),
                                      order.index(0), meta={"idx": i}), r["question"]))
    return out


def commonsense_qa(raw):
    df = real._parquets(os.path.join(raw, "tau__commonsense_qa/data/validation-*.parquet"))
    return [real.item("commonsense_qa", "test", None, None, jstate({"question": r["question"].strip()}), ANSWER_Q,
                      letter_options(list(r["choices"]["text"])), list(r["choices"]["label"]).index(r["answerKey"]),
                      meta={"id": r["id"]}) for r in df.to_dict("records")]


CNLI = ["entailed: The agreement states or clearly implies this", "contradicted: The agreement states the opposite",
        "not mentioned: The agreement does not address this"]


def contract_nli(raw):
    d = json.load(open(os.path.join(raw, "stanfordnlp__contract-nli/contract-nli/test.json")))
    idx = {"Entailment": 0, "Contradiction": 1, "NotMentioned": 2}
    return [real.item("contract_nli", "test", None, None, doc["text"],
                      'Does the agreement entail, contradict, or not mention this statement: "%s"'
                      % d["labels"][key]["hypothesis"], CNLI, idx[ann["choice"]],
                      meta={"doc_id": doc["id"], "hypothesis": key})
            for doc in d["documents"] for key, ann in doc["annotation_sets"][0]["annotations"].items()]


def quality(raw):
    path = os.path.join(raw, "nyu-mll__quality/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev")
    out = []
    for art in (json.loads(l) for l in open(path)):
        for q in art["questions"]:
            opts = [o.strip() for o in q["options"]]
            if len(set(opts)) == len(opts) and q.get("gold_label"):
                out.append(real.item("quality", "test", None, None, art["article"].strip(),
                                     q["question"].strip() + " (Answer from the article above.)", letter_options(opts),
                                     int(q["gold_label"]) - 1,
                                     meta={"article_id": art["article_id"], "question_id": q["question_unique_id"],
                                           "difficult": int(q.get("difficult", 0))}))
    return out


SYS1CAL_Q = ("The state fully specifies a probability problem. Treat all numeric probabilities, counts, ratios, tables, "
             "filters, and likelihoods in the state as authoritative. Ignore fields marked irrelevant or distractor. "
             "Return the probability distribution for the random outcome or latent hypothesis described by the "
             "benchmark question; do not report confidence in your reasoning. ")
SYS1CAL_CRITERIA = {   # the Sys1Cal authors' Jev adapter (src/jev_prob_bench/adapters/jev.py, _choice_criteria)
    "A": "The realized outcome is A, or the queried item belongs to class A.",
    "not_A": "The realized outcome is not A, or the queried item does not belong to class A.",
    "H": "The latent hypothesis H is true after conditioning on the supplied evidence.",
    "not_H": "The latent hypothesis H is false after conditioning on the supplied evidence.",
    "target": "The compound event named in the question occurs.",
    "not_target": "The compound event named in the question does not occur.",
}


def sys1cal(raw):
    """Sys1Cal-v1 v0.1.0 (all 365 items), posed as its authors pose it to Jev; the target is the exact probability."""
    path = os.path.join(raw, "little-g-ai__Sys1Cal-v1/datasets/generated/v0.1.0_tiny_cleanvars_binary.jsonl")
    out = []
    for r in (json.loads(l) for l in open(path)):
        crit = {o: SYS1CAL_CRITERIA.get(o, "The realized outcome, class, or latent hypothesis is exactly %s." % o)
                for o in r["outcomes"]}
        out.append(real.item("sys1cal", "test", int(r["difficulty"]), "sys1cal_difficulty", real.typed_state(r["state"]),
                             SYS1CAL_Q + r["prompt"], ["%s: %s" % (o, crit[o]) for o in r["outcomes"]], None,
                             dist=[float(r["gold_distribution"][o]) for o in r["outcomes"]],
                             meta={"instance_id": r["instance_id"], "latent_instance_id": r["latent_instance_id"],
                                   "family": r["family"], "representation": r["representation"],
                                   "exact_probability": True}))
    return out


# ---------------------------------------------------------------- build

NEAR = 0.8


class TrainOverlap:
    """Test candidates that repeat training material: the same state once case and punctuation are dropped, or a
    sentence set with Jaccard overlap >= NEAR with some training state (the criteria of data.check_overlap)."""

    def __init__(self, train_items):
        self.states = {norm(it["state"]) for it in train_items}
        sets = [sentence_set(it["state"]) for it in train_items]
        post = collections.defaultdict(list)
        for j, st in enumerate(sets):
            for x in st:
                post[x].append(j)
        self.post = {x: np.asarray(v, dtype=np.int64) for x, v in post.items()}
        self.sizes = np.asarray([len(st) for st in sets], dtype=np.float64)

    def ok(self, it):
        if norm(it["state"]) in self.states:
            return False
        st = sentence_set(it["state"])
        lists = [self.post[x] for x in st if x in self.post]
        if not lists:
            return True
        inter = np.bincount(np.concatenate(lists), minlength=len(self.sizes))
        nz = np.nonzero(inter)[0]
        return float((inter[nz] / (len(st) + self.sizes[nz] - inter[nz])).max()) < NEAR

    def filter(self, items, name):
        keep = [it for it in items if self.ok(it)]
        if len(keep) < len(items):
            note("%s: %d of %d test candidates repeat training material (same state or >= %.1f of its sentences), "
                 "left out" % (name, len(items) - len(keep), len(items), NEAR))
        return keep


def build(raw, pw_excluded):
    """The training items T and the test candidates X, each {source: items}. pw_excluded: ids of ProofWriter test questions
    that are left out of the test candidates (--exclude-ids)."""
    T, X = {}, {}
    names = real.clutrr_names(raw)

    def story_key(it):
        it["meta"]["story_key"] = fingerprint(real.clutrr_renamed(it, names))
        return it

    # ================================================================ training
    kv = kev_groups(raw)
    KEV = [(s, "A", 300, 50) for s in ("agnews", "dbpedia14", "amazon", "yelp", "imdb", "sst5", "banking77", "trec")] + \
          [("compositional", "B", 1110, 150), ("legacy_policy", "B", 690, 150), ("mnli", "E", 800, 100),
           ("boolq", "E", 800, 100)]
    kev_used = {}
    for src, layer, n_tr, _ in KEV:
        tr, used = kev_take(kv[src], n_tr, set(), lambda it: True, src, "train")
        T["kev_" + src] = tag(tr, layer, "train", "kev_" + src, kev_group)
        kev_used[src] = set(used)
    n_comp = len(unique(it for g in kv["compositional"].values() for it in g))
    note("Kev decision-v7 compositional: %d questions, %d distinct -> training 1,110, in-distribution test 150; "
         "policies make up the difference (690 instead of 600)" %
         (sum(len(g) for g in kv["compositional"].values()), n_comp))

    pw_tr = unique(real.proofwriter(raw, "train", ["depth-5"], 10 ** 9, SEED, max_qdep=5))
    T["proofwriter"] = tag(stratified(pw_tr, [K, ANS], 1500, "pw_tr"), "B", "train", "proofwriter")
    cl_tr_all = [story_key(it) for it in unique(it for it in real.clutrr(raw, "train") if it["k"] in (2, 3, 4))]
    T["clutrr"] = tag(stratified(cl_tr_all, [K], 1200, "cl_tr"), "B", "train", "clutrr")

    # MuSiQue: disjoint questions for B, C and D; training from the train file, tests from the dev file
    rows = {s: real._musique_rows(raw, "ans", s) for s in ("train", "dev")}
    pairs = {}
    for s in ("train", "dev"):
        by = collections.defaultdict(dict)
        for r in real._musique_rows(raw, "full", s):
            by[r["id"]][bool(r["answerable"])] = r
        pairs[s] = {i: (v[True], v[False]) for i, v in by.items() if True in v and False in v}
    pool = [r["answer"] for r in rows["train"] + rows["dev"] if r["answer"].strip()]
    hop = lambda r: real._hops(r["id"])
    tr23 = [r for r in rows["train"] if hop(r) in (2, 3) and r["answer"].strip()]
    c_rows = stratified([r for r in tr23 if r["id"] in pairs["train"]], [hop], 560, "mq_c_tr")
    c_pairs = pairs_by_hop(c_rows, pairs["train"], pool, "train", 500, "mq_c_tr")
    left = [r for r in tr23 if r["id"] not in {r["id"] for r in c_rows}]
    b_rows = stratified(left, [hop], 1000, "mq_b_tr")
    left = [r for r in left if r["id"] not in {r["id"] for r in b_rows}]
    d_rows = stratified(left, [hop], 600, "mq_d_tr")
    T["musique"] = tag([musique_short(r, pool, "musique_choice", "train") for r in b_rows], "B", "train", "musique")
    T["musique_pairs"] = tag([x for p in c_pairs for x in p], "C", "train", "musique_pairs",
                             lambda it: it["meta"]["pair_id"])
    T["musique_long"] = tag([musique_lengths(r, pool, "musique_long", "train",
                                             ["medium" if rng_for("mq_d_v", r["id"]).random() < 0.5 else "long"])[0]
                             for r in d_rows], "D", "train", "musique_long", lambda it: it["meta"]["id"])
    trained_mq = b_rows + d_rows + [pairs["train"][p[0]["meta"]["id"]][0] for p in c_pairs]

    hp_tr = sample(hotpot_rows(raw, "train"), 400, "hp_tr")
    T["hotpotqa"] = tag([x for r in hp_tr for x in hotpot_lengths(r, "train", ["long"])], "D", "train", "hotpotqa")
    sq = squad(raw, "train")
    T["squad2"] = tag(sample(sq[False], 400, "sq_tr_a") + sample(sq[True], 400, "sq_tr_u"), "C", "train", "squad2")
    ch = unique(real.chaosnli(raw, ["snli", "alphanli"]))
    src_of = lambda it: it["meta"]["source"]
    T["chaosnli"] = tag(stratified(ch, [src_of], 500, "chaos_tr"), "C", "train", "chaosnli")

    ov = TrainOverlap([it for v in T.values() for it in v])

    # ================================================================ tests (candidates checked against training)
    for src, layer, _, n_te in KEV:
        te, _ = kev_take(kv[src], n_te, kev_used[src], ov.ok, src, "test")
        X["kev_%s_id" % src] = tag(te, layer, "in_dist", "kev_%s_id" % src, kev_group)

    # Kev transfer-v9 development, clean variants (never evaluated before): far transfer
    v9 = kev_v9(raw)
    for src, layer in (("emotion", "A"), ("tweet_offensive", "A"), ("composition_holdout", "B"), ("legacy_holdout", "B"),
                       ("buried", "D"), ("qnli", "E"), ("paws", "E"), ("mmlu", "F"), ("mmlu_pro", "F"), ("sciq", "F")):
        X["kev_v9_" + src] = tag(ov.filter(unique(v9[src]), "kev_v9_" + src), layer, "far", "kev_v9_" + src, kev_group)
        if len(X["kev_v9_" + src]) < len(v9[src]):
            note("Kev v9 dev %s: %d questions, %d kept" % (src, len(v9[src]), len(X["kev_v9_" + src])))
    un = collections.defaultdict(list)
    for it in kev_unknowable_pairs(v9):
        un[it["meta"]["pair_id"]].append(it)
    bad = [g for g, v in un.items() if not all(ov.ok(it) for it in v)]
    note("Kev v9 dev unknowable pairs: %d groups repeat training material (an unknowable state is a training case "
         "with the deciding sentence removed), left out" % len(bad))
    X["kev_v9_unknowable_pairs"] = tag([it for g, v in un.items() if g not in bad for it in v], "C", "far",
                                       "kev_v9_unknowable_pairs", lambda it: it["meta"]["pair_id"])
    kev_blob = "\n".join(norm(it["state"] + " " + it["question"]) for v in v9.values() for it in v)

    def pool_of(items, name, kev=False):
        items = ov.filter(unique(items), name)
        if kev:
            keep = [it for it in items if len(it["meta"].get("text_key", "")) < 20 or it["meta"]["text_key"] not in kev_blob]
            note("%s: %d of %d candidates are Kev v9 dev examples, left out" % (name, len(items) - len(keep), len(items)))
            items = keep
        return items

    # A far
    X["emotion"] = tag(sample(pool_of(emotion(raw), "emotion", True), 300, "emotion"), "A", "far", "emotion")
    X["tweet_eval"] = tag(stratified(pool_of(tweet_eval(raw), "tweet_eval", True), [lambda it: it["meta"]["task"]], 300,
                                     "tweet"), "A", "far", "tweet_eval")
    X["yahoo_answers"] = tag(sample(pool_of(yahoo(raw), "yahoo_answers"), 200, "yahoo"), "A", "far", "yahoo_answers")

    # B: ProofWriter (depth 0-5; tests only from test questions that are not listed in --exclude-ids), CLUTRR
    note("ProofWriter test questions listed in --exclude-ids, left out:", len(pw_excluded))
    pw_te = pool_of([it for it in real.proofwriter(raw, "test", ["depth-5"], 10 ** 9, SEED, max_qdep=5)
                     if it["meta"]["id"] not in pw_excluded], "proofwriter_id")
    X["proofwriter_id"] = tag(stratified(pw_te, [K, ANS], 300, "pw_te"), "B", "in_dist", "proofwriter_id")
    nl = pool_of([it for it in real.proofwriter(raw, "test", ["NatLang"], 10 ** 9, SEED, max_qdep=5)
                  if it["meta"]["id"] not in pw_excluded], "proofwriter_natlang")
    X["proofwriter_natlang"] = tag(stratified(nl, [K, ANS], 400, "pw_nl"), "B", "near", "proofwriter_natlang")
    train_stories = {it["meta"]["story_key"] for it in cl_tr_all}
    cl_te = [story_key(it) for it in pool_of(real.clutrr(raw, "test"), "clutrr")]
    cl_ok = [it for it in cl_te if it["meta"]["story_key"] not in train_stories]
    note("CLUTRR test stories equal to a training story up to names, left out:", len(cl_te) - len(cl_ok))
    X["clutrr_id"] = tag(stratified([it for it in cl_ok if it["k"] <= 4], [K], 200, "cl_id"), "B", "in_dist", "clutrr_id")
    X["clutrr_long"] = tag(stratified([it for it in cl_ok if it["k"] >= 5], [K], 600, "cl_long"), "B", "near",
                           "clutrr_long")

    # MuSiQue tests: dev questions sharing no single-hop sub-question with a training question, and none of whose
    # test items repeats training material
    trained_subq = set().union(*(subq_ids(r) for r in trained_mq))
    trained_pars = {p["paragraph_text"] for r in trained_mq for p in r["paragraphs"] if p.get("is_supporting")}
    dev = [r for r in rows["dev"] if r["answer"].strip() and r["id"] in pairs["dev"]]
    dev_ok = [r for r in dev if not subq_ids(r) & trained_subq]
    note("MuSiQue dev questions sharing a single-hop sub-question with a training question, left out: %d of %d"
         % (len(dev) - len(dev_ok), len(dev)))
    note("MuSiQue dev questions with a supporting paragraph that supports a training question:",
         sum(any(p.get("is_supporting") and p["paragraph_text"] in trained_pars for p in r["paragraphs"]) for r in dev_ok))
    built = {}
    for r in dev_ok:
        pr = musique_pair(*pairs["dev"][r["id"]], pool, "musique_pair", "test")
        short = musique_short(r, pool, "musique_choice", "test")
        lengths = musique_lengths(r, pool, "musique_long", "test", ["short", "medium", "long"])
        built[r["id"]] = {"pair": pr if pr and all(ov.ok(x) for x in pr) else None,
                          "short": short if ov.ok(short) else None,
                          "lengths": lengths if all(ov.ok(x) for x in lengths) else None}
    four = sample([r for r in dev_ok if hop(r) == 4], 10 ** 9, "mq_4")
    c4 = [built[r["id"]]["pair"] for r in four if built[r["id"]]["pair"]][:150]
    c4_ids = {p[0]["meta"]["id"] for p in c4}
    b4 = [built[r["id"]]["short"] for r in four if r["id"] not in c4_ids and built[r["id"]]["short"]][:300]
    note("MuSiQue dev 4-hop questions available: %d -> near-transfer pairs %d, near-transfer choice %d"
         % (len(four), len(c4), len(b4)))
    X["musique_pairs_4hop"] = tag([x for p in c4 for x in p], "C", "near", "musique_pairs_4hop",
                                  lambda it: it["meta"]["pair_id"])
    X["musique_4hop"] = tag(b4, "B", "near", "musique_4hop")
    d23 = [r for r in dev_ok if hop(r) in (2, 3)]
    c23 = stratified([built[r["id"]]["pair"] for r in d23 if built[r["id"]]["pair"]], [lambda p: p[0]["k"]], 100,
                     "mq_c_te")
    taken = {p[0]["meta"]["id"] for p in c23}
    b23 = stratified([built[r["id"]]["short"] for r in d23 if r["id"] not in taken and built[r["id"]]["short"]], [K],
                     200, "mq_b_te")
    taken |= {it["meta"]["id"] for it in b23}
    dl = stratified([built[r["id"]]["lengths"] for r in d23 if r["id"] not in taken and built[r["id"]]["lengths"]],
                    [lambda v: v[0]["k"]], 100, "mq_d_te")
    X["musique_pairs_id"] = tag([x for p in c23 for x in p], "C", "in_dist", "musique_pairs_id",
                                lambda it: it["meta"]["pair_id"])
    X["musique_id"] = tag(b23, "B", "in_dist", "musique_id")
    X["musique_lengths"] = tag([x for v in dl for x in v], "D", "in_dist", "musique_lengths", lambda it: it["meta"]["id"])

    # D: HotpotQA at three lengths (all three versions must be clear of training material)
    hp_te = [v for v in (hotpot_lengths(r, "test", ["short", "medium", "long"]) for r in hotpot_rows(raw, "test"))
             if all(ov.ok(x) for x in v)]
    X["hotpotqa_lengths"] = tag([x for v in sample(hp_te, 50, "hp_te") for x in v], "D", "in_dist", "hotpotqa_lengths",
                                lambda it: it["meta"]["id"])

    # B far (HoVer claims were written from HotpotQA questions: leave out those from our HotpotQA training questions)
    X["folio"] = tag(sample(pool_of(real.folio(raw, "v2"), "folio"), 400, "folio"), "B", "far", "folio")
    bb = [it for it in real.bbh(raw) if it["dataset"] in ("bbh_tracking", "bbh_logical_deduction")]
    X["bbh"] = tag(stratified(pool_of(bb, "bbh"), [lambda it: (it["dataset"], it["k"])], 600, "bbh"), "B", "far", "bbh")
    hpqa = {x["uid"]: x.get("hpqa_id") for x in
            json.load(open(os.path.join(raw, "hover-nlp__hover/data/hover_train_release_v1.1.json")))}
    hp_ids = {r["id"] for r in hp_tr}
    hv = pool_of(real.hover(raw, "train", {2, 3, 4}), "hover")
    for it in hv:
        it["meta"]["hpqa_id"] = hpqa.get(it["meta"]["uid"])
    hv_ok = [it for it in hv if it["meta"]["hpqa_id"] not in hp_ids]
    note("HoVer claims written from one of our HotpotQA training questions, left out:", len(hv) - len(hv_ok))
    X["hover"] = tag(stratified(hv_ok, [K], 450, "hover"), "B", "far", "hover")

    # C: SQuAD 2.0, ChaosNLI, Sys1Cal
    sq = squad(raw, "dev")
    X["squad2_id"] = tag(sample(pool_of(sq[False], "squad2_id answerable"), 150, "sq_te_a") +
                         sample(pool_of(sq[True], "squad2_id unanswerable"), 150, "sq_te_u"), "C", "in_dist", "squad2_id")
    ch_tr = {it["fingerprint"] for it in T["chaosnli"]}
    X["chaosnli_id"] = tag(stratified(pool_of([it for it in ch if it["fingerprint"] not in ch_tr], "chaosnli_id"),
                                      [src_of], 100, "chaos_te"), "C", "in_dist", "chaosnli_id")
    X["chaosnli_mnli"] = tag(sample(pool_of(real.chaosnli(raw, ["mnli"]), "chaosnli_mnli"), 500, "chaos_m"), "C", "near",
                             "chaosnli_mnli")
    X["sys1cal"] = tag(pool_of(sys1cal(raw), "sys1cal"), "C", "far", "sys1cal", lambda it: it["meta"]["latent_instance_id"])

    # D far
    X["contract_nli"] = tag(sample(pool_of(contract_nli(raw), "contract_nli"), 300, "cnli"), "D", "far", "contract_nli",
                            lambda it: it["meta"]["doc_id"])
    X["quality"] = tag(sample(pool_of(quality(raw), "quality"), 300, "quality"), "D", "far", "quality",
                       lambda it: it["meta"]["article_id"])

    # E near and far
    X["mnli_mismatched"] = tag(sample(pool_of(mnli_mismatched(raw), "mnli_mismatched"), 300, "mnli_mm"), "E", "near",
                               "mnli_mismatched")
    X["anli_dev"] = tag(stratified(pool_of(anli_dev(raw), "anli_dev"), [K], 300, "anli"), "E", "far", "anli_dev")
    X["qnli"] = tag(sample(pool_of(qnli(raw), "qnli", True), 250, "qnli"), "E", "far", "qnli")
    X["paws"] = tag(sample(pool_of(paws(raw), "paws", True), 250, "paws"), "E", "far", "paws")
    X["wanli"] = tag(sample(pool_of(wanli(raw), "wanli"), 250, "wanli"), "E", "far", "wanli")

    # F knowledge, test only
    X["sciq"] = tag(sample(pool_of(sciq(raw), "sciq", True), 150, "sciq"), "F", "far", "sciq")
    X["arc_easy"] = tag(sample(pool_of(arc(raw, "ARC-Easy"), "arc_easy"), 150, "arc_e"), "F", "far", "arc_easy")
    X["arc_challenge"] = tag(sample(pool_of(arc(raw, "ARC-Challenge"), "arc_challenge"), 400, "arc_c"), "F", "far",
                             "arc_challenge")
    X["mmlu"] = tag(stratified(pool_of(mmlu(raw), "mmlu", True), [lambda it: it["meta"]["mmlu_group"],
                                                                  lambda it: it["meta"]["subject"]], 600, "mmlu"),
                    "F", "far", "mmlu")
    X["mmlu_pro"] = tag(sample(pool_of(mmlu_pro(raw), "mmlu_pro", True), 300, "mmlu_pro"), "F", "far", "mmlu_pro")
    X["commonsense_qa"] = tag(sample(pool_of(commonsense_qa(raw), "commonsense_qa"), 300, "csqa"), "F", "far",
                              "commonsense_qa")
    return T, X


def assign_dev(X):
    """Dev (20% of each source) / final split. Test items are first joined into components: items of one group (a
    question with its versions, a pair, a Kev group, a document), items with the same state, CLUTRR stories that differ
    only in names, MuSiQue questions sharing a single-hop sub-question, HoVer claims from the same HotpotQA question,
    and items sharing >= NEAR of their sentences, across all sources. Components are visited in an order fixed by the
    seed and never split; a component goes to dev when that brings the sources it touches closer, in total, to 20% dev
    (for a component inside one source: while that source's dev part is below 20%). meta.group becomes the component,
    the unit for bootstrap resampling."""
    items = [(name, it) for name in sorted(X) for it in X[name]]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    first = {}
    for i, (name, it) in enumerate(items):
        m = it["meta"]
        keys = [("group", name, m["group"]), ("state", norm(it["state"]))]
        keys += [("story", m["story_key"])] if m.get("story_key") else []
        keys += [("subq", q) for q in m.get("subq", [])]
        keys += [("hpqa", m["hpqa_id"])] if m.get("hpqa_id") else []
        for k in keys:
            if k in first:
                union(i, first[k])
            else:
                first[k] = i
    sets = [sentence_set(it["state"]) for _, it in items]
    post = collections.defaultdict(list)
    for i, st in enumerate(sets):
        for x in st:
            post[x].append(i)
    for i, st in enumerate(sets):
        inter = collections.Counter(j for x in st for j in post[x] if j < i)
        for j, c in inter.items():
            if c / (len(st) + len(sets[j]) - c) >= NEAR:
                union(i, j)
    label = {}
    for i, (name, it) in enumerate(items):
        r = find(i)
        label.setdefault(r, "%s/%s" % (items[r][0], items[r][1]["meta"]["group"]))
        it["meta"]["group"] = label[r]
    sizes = collections.Counter(it["meta"]["group"] for _, it in items)
    note("dev / final components: %d items in %d components, largest %d" % (len(items), len(sizes),
                                                                           max(sizes.values())))
    comps = collections.defaultdict(list)
    for name, it in items:
        comps[it["meta"]["group"]].append((name, it))
    size = collections.Counter(name for name, _ in items)
    n_dev = collections.Counter()
    D, F = {name: [] for name in X}, {name: [] for name in X}
    for g in sorted(comps, key=lambda g: hashlib.sha1(("%d/%s" % (SEED, g)).encode()).hexdigest()):
        cnt = collections.Counter(name for name, _ in comps[g])
        gap = lambda extra: sum(abs(n_dev[s] + extra * n - DEV_SHARE * size[s]) for s, n in cnt.items())
        to_dev = gap(1) < gap(0)
        if to_dev:
            n_dev.update(cnt)
        for name, it in comps[g]:
            (D if to_dev else F)[name].append(it)
    return D, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="the folder with the raw datasets (layout: data/README.md)")
    ap.add_argument("--out", default="datasets/main")
    ap.add_argument("--tokenizer", required=True, help="the folder of Ouro-1.4B: its tokenizer counts meta.tokens")
    ap.add_argument("--exclude-ids", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "proofwriter_excluded_ids.json"),
                    help="a JSON list of ProofWriter test question ids that are left out of the test candidates "
                         "(default: the list the main set of the paper was built with; '' for none)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from sansi.model import render_prompt
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    pw_excluded = set(json.load(open(args.exclude_ids))) if args.exclude_ids else set()
    T, X = build(args.raw, pw_excluded)
    JB = {"jevbench_easy": real.jevbench(args.raw, "easy"), "jevbench_standard": real.jevbench(args.raw, "original"),
          "jevbench_hard": real.jevbench(args.raw, "hard")}
    for k, v in JB.items():
        tag(v, "JB", "jevbench", k)
    fps = collections.Counter(it["fingerprint"] for d in (T, X, JB) for v in d.values() for it in v)
    note("items with the same text as another item:", sum(c - 1 for c in fps.values() if c > 1))
    assert all(c == 1 for c in fps.values())
    D, F = assign_dev(X)

    files, rows = {}, []

    def write(split, name, items):
        for it in items:
            it["meta"]["tokens"] = len(tok.encode(render_prompt(it), add_special_tokens=False))
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        write_jsonl(items, os.path.join(args.out, split, name + ".jsonl.gz"))
        toks = sorted(it["meta"]["tokens"] for it in items)
        info = {"n": len(items), "layer": items[0]["meta"]["layer"], "tier": items[0]["meta"]["tier"],
                "groups": len({it["meta"]["group"] for it in items}),
                "unanswerable": sum(bool(it["meta"].get("unanswerable")) for it in items),
                "soft": sum(not it["deterministic"] and not it["meta"].get("unanswerable") for it in items),
                "k": dict(sorted(collections.Counter(str(it["k"]) for it in items).items())),
                "n_options": dict(sorted(collections.Counter(len(it["options"]) for it in items).items())),
                "gold_index": dict(sorted(collections.Counter(str(it["answer_index"]) for it in items).items())),
                "tokens_median": toks[len(toks) // 2], "tokens_p95": toks[int(0.95 * (len(toks) - 1))],
                "tokens_max": toks[-1]}
        files["%s/%s" % (split, name)] = info
        rows.append((split, name, info))

    for name, items in sorted(T.items()):
        write("train", name, items)
    for name in sorted(X):
        write("dev", name, D[name])
        write("test", name, F[name])
    for name, items in JB.items():
        write("test", name, items)

    totals = collections.defaultdict(collections.Counter)
    for split, name, info in rows:
        totals[split]["%s/%s" % (info["layer"], info["tier"])] += info["n"]
    manifest = {"spec": "SanSi main set (data/build_main.py)", "seed": SEED, "dev_share": DEV_SHARE,
                "near_dup": NEAR, "args": vars(args), "tokenizer": args.tokenizer, "files": files,
                "totals": {s: dict(sorted(c.items())) for s, c in totals.items()}, "build_notes": REPORT}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    lines = ["%-5s %-28s %-2s %-8s %6d  groups %5d  unans %4d  soft %4d  tokens med %5d p95 %6d max %6d" % (
        split, name, info["layer"], info["tier"], info["n"], info["groups"], info["unanswerable"], info["soft"],
        info["tokens_median"], info["tokens_p95"], info["tokens_max"]) for split, name, info in rows]
    lines += ["%s %d %s" % (s, sum(c.values()), dict(sorted(c.items()))) for s, c in totals.items()]
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(REPORT + [""] + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
