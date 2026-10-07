"""Converters from public datasets to the item format of the main set.

Every converter returns items with the same fields:

  dataset, split, k (depth, or None), depth_kind, qtype ("choice" | "noul" | "score"),
  state, question, options, answer_index, answer_distribution, deterministic,
  fingerprint, meta

`k` is the dataset's own depth annotation (ProofWriter QDep, CLUTRR relation-chain
length, MuSiQue / HoVer hop count, BBH number of objects).
Soft-label items (ChaosNLI) carry the human label distribution and deterministic=False.

Raw files are read from the folder given as `raw`, laid out as downloaded by
huggingface_hub.snapshot_download(local_dir=<owner>__<name>) plus the two official
files fetched separately (CLUTRR split csvs, HoVer release json). data/README.md lists every file that is read.
"""
import glob
import json
import os
import random
import re

from .common import fingerprint

NLI_OPTIONS = ["entailment", "neutral", "contradiction"]


def item(dataset, split, k, depth_kind, state, question, options, answer_index=None, dist=None,
         qtype="choice", meta=None):
    options = list(options)
    assert len(set(options)) == len(options), (dataset, options)
    if dist is None:
        dist = [0.0] * len(options)
        dist[answer_index] = 1.0
    deterministic = max(dist) == 1.0
    if answer_index is None:
        answer_index = max(range(len(dist)), key=lambda i: dist[i])
    return {
        "dataset": dataset, "family": dataset, "split": split, "k": k, "depth_kind": depth_kind,
        "qtype": qtype, "state": state.strip(), "question": question.strip(), "options": options,
        "answer_index": answer_index, "answer_distribution": [round(float(p), 6) for p in dist],
        "deterministic": deterministic,
        "fingerprint": fingerprint(dataset, state.strip(), question.strip(), options),
        "meta": meta or {},
    }


def _parquets(path_glob):
    import pandas as pd
    files = sorted(glob.glob(path_glob))
    assert files, path_glob
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _balanced(rows, key, per_group, rng):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for g in sorted(groups, key=str):
        rs = groups[g]
        rng.shuffle(rs)
        out += rs[:per_group]
    return out


# ---------------------------------------------------------------- ProofWriter (OWA)

def proofwriter(raw, split, configs, per_group, seed, max_qdep=None):
    """True / False / Unknown under the open-world assumption. k = QDep (proof depth)."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(raw, "tasksource__proofwriter/data/%s-*.parquet" % split)))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.config.isin(configs)]
    if max_qdep is not None:
        df = df[df.QDep.astype(int) <= max_qdep]
    rows = df.to_dict("records")
    rng = random.Random(seed)
    rows = _balanced(rows, lambda r: (r["config"], int(r["QDep"]), r["answer"]), per_group, rng)
    amap = {"True": 0, "False": 1, "Unknown": 2}
    out = []
    for r in rows:
        q = 'Using only the facts and rules above, is the statement "%s" true, false, or unknown?' % r["question"].strip()
        out.append(item("proofwriter_" + r["config"].replace("depth-", "d"), split, int(r["QDep"]), "proof_depth",
                        r["theory"], q, ["true", "false", "unknown"], amap[r["answer"]],
                        meta={"id": r["id"], "config": r["config"], "NFact": int(r["NFact"]), "NRule": int(r["NRule"])}))
    return out


def folio(raw, version="v2"):
    """FOLIO (expert-written first-order-logic stories), asked exactly like ProofWriter; test only, no depth.

    Both official files (train and validation) become test items. Labels True / False / Uncertain, mapped to
    ProofWriter's true / false / unknown. version "v2" is the authors' corrected release (Hugging Face
    yale-nlp/FOLIO, gated); "v0.0" is the first release on GitHub."""
    amap = {"True": 0, "False": 1, "Unknown": 2, "Uncertain": 2}
    out = []
    for official in ("train", "validation"):
        if version == "v2":
            path = os.path.join(raw, "yale-nlp__FOLIO", "folio_v2_%s.jsonl" % official)
        else:
            path = os.path.join(raw, "Yale-LILY__FOLIO/data", version, "folio-%s.jsonl" % official)
        for i, r in enumerate(json.loads(l) for l in open(path) if l.strip()):
            premises = r["premises"] if isinstance(r["premises"], list) else r["premises"].split("\n")
            premises = [p.strip() for p in premises if p.strip()]
            theory = " ".join(p if p.endswith((".", "!", "?")) else p + "." for p in premises)
            q = 'Using only the facts and rules above, is the statement "%s" true, false, or unknown?' % r["conclusion"].strip()
            out.append(item("folio", "test", None, None, theory, q, ["true", "false", "unknown"], amap[r["label"]],
                            meta={"official_split": official, "idx": i, "story_id": r.get("story_id"),
                                  "example_id": r.get("example_id"), "version": version}))
    return out


# ---------------------------------------------------------------- CLUTRR

def clutrr(raw, split, task="gen_train234_test2to10"):
    """Kinship relation between two people; k = number of relations in the chain."""
    import pandas as pd
    base = os.path.join(raw, "CLUTRR__v1/data", task)
    rel = {}
    for s in ("train", "validation", "test"):
        d = pd.read_csv(os.path.join(base, "%s.csv" % s))
        for t, tt in zip(d.target, d.target_text):
            rel[int(t)] = tt
    options = [rel[i] for i in sorted(rel)]
    d = pd.read_csv(os.path.join(base, "%s.csv" % split))
    out = []
    for r in d.to_dict("records"):
        a, b = re.findall(r"'([^']+)'", r["query"])
        story = r["story"].replace("[", "").replace("]", "")
        k = int(str(r["task_name"]).split(".")[1])
        q = "In this story, what is %s to %s? Complete: %s is %s's ___." % (b, a, b, a)
        out.append(item("clutrr", split, k, "relation_length", story, q, options, options.index(r["target_text"]),
                        meta={"id": r["id"], "f_comb": r["f_comb"], "task": task}))
    return out


def clutrr_names(raw, task="gen_train234_test2to10"):
    """(split, id) -> the people in that CLUTRR story, read from the official csv files."""
    import pandas as pd
    names = {}
    for split in ("train", "validation", "test"):
        d = pd.read_csv(os.path.join(raw, "CLUTRR__v1/data", task, "%s.csv" % split))
        for i, g in zip(d.id, d.genders):
            names[(split, i)] = [x.split(":")[0] for x in g.split(",")]
    return names


def clutrr_renamed(it, names):
    """Story and question with the people's names replaced by P0, P1, ... in order of first appearance.

    CLUTRR fills crowd-written templates with names, so two items can be the same story with different
    people; they share this key."""
    text = it["state"] + " || " + it["question"]
    found = []
    for n in set(names[(it["split"], it["meta"]["id"])]):
        m = re.search(r"\b%s\b" % re.escape(n), text)
        if m:
            found.append((m.start(), n))
    for j, (_, n) in enumerate(sorted(found)):
        text = re.sub(r"\b%s\b" % re.escape(n), "P%d" % j, text)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


# ---------------------------------------------------------------- MuSiQue

def _musique_rows(raw, version, split):
    path = os.path.join(raw, "bdsaglam__musique/musique_%s_v1.0_%s.jsonl" % (version, split))
    return [json.loads(l) for l in open(path)]


def _hops(qid):
    return int(qid.split("hop")[0])


# ---------------------------------------------------------------- HoVer

def hover(raw, split, hops):
    """Claim verification from gold evidence; k = number of hops (official annotation).
    Evidence text comes from the Dzeniks/hover mirror, joined to the official release by claim."""
    off_split = {"train": "train", "test": "dev"}[split]
    official = json.load(open(os.path.join(raw, "hover-nlp__hover/data/hover_%s_release_v1.1.json" % off_split)))
    idx = {}
    for x in official:
        idx.setdefault(x["claim"].strip(), []).append(x)
    dz = [json.loads(l) for l in open(os.path.join(raw, "Dzeniks__hover/%s.jsonl" % split))]
    out, dropped = [], 0
    for y in dz:
        cands = idx.get(y["claim"].strip(), [])
        lab = y["label"]                       # 0 = SUPPORTED in the mirror
        cands = [c for c in cands if {"SUPPORTED": 0, "NOT_SUPPORTED": 1}[c["label"]] == lab]
        if len({c["num_hops"] for c in cands}) != 1:
            dropped += 1
            continue
        h = cands[0]["num_hops"]
        if h not in hops:
            continue
        evidence = " ".join(s.strip() for s in y["evidence"].split("\n") if s.strip())
        q = 'Is this claim supported by the evidence above? Claim: "%s"' % y["claim"].strip()
        out.append(item("hover", split, h, "hops", evidence, q, ["supported", "not supported"], lab,
                        meta={"uid": cands[0]["uid"]}))
    return out


# ---------------------------------------------------------------- BIG-Bench Hard (test only)

def bbh(raw):
    out = []
    size = {"three": 3, "five": 5, "seven": 7}
    for task_dir in sorted(glob.glob(os.path.join(raw, "lukaemon__bbh/*"))):
        task = os.path.basename(task_dir)
        files = glob.glob(os.path.join(task_dir, "*.parquet"))
        if not files:
            continue
        import pandas as pd
        df = pd.read_parquet(files[0])
        for i, r in enumerate(df.to_dict("records")):
            text, target = r["input"].strip(), r["target"].strip()
            if task.startswith(("tracking_shuffled_objects", "logical_deduction")):
                body, opt_text = text.split("Options:")
                opts = re.findall(r"\(([A-Z])\)\s*(.+)", opt_text)
                letters = [l for l, _ in opts]
                options = [o.strip() for _, o in opts]
                k = size[task.split("_")[-2]]
                name = "bbh_tracking" if task.startswith("tracking") else "bbh_logical_deduction"
                q = ("Which option correctly completes the last sentence?" if name == "bbh_tracking"
                     else "Which option is correct?")
                out.append(item(name, "test", k, "objects", body, q, options, letters.index(target.strip("()")),
                                meta={"task": task, "idx": i}))
            elif task == "web_of_lies":
                body = text.replace("Question:", "").strip()
                m = re.search(r"(Does .+\?)$", body)
                statements, q = body[: m.start()].strip(), m.group(1)
                out.append(item("bbh_web_of_lies", "test", len(re.findall(r"\.", statements)), "statements",
                                statements, q, ["yes", "no"], 0 if target == "Yes" else 1, meta={"idx": i}))
            elif task == "boolean_expressions":
                expr = re.sub(r"\s+is$", "", text)
                out.append(item("bbh_boolean_expressions", "test", len(re.findall(r"\b(and|or|not)\b", expr)),
                                "operators", "Expression: " + expr, "What does this Boolean expression evaluate to?",
                                ["True", "False"], 0 if target == "True" else 1, meta={"idx": i}))
    return out


# ---------------------------------------------------------------- Jev-style typed decisions: JevBench, Kev

def typed_options(qtype, labels, criteria):
    """One option per label, in the given order, written "label: definition" with the rubric's definition of that
    label: noul "no: <criteria.false>", "yes: <criteria.true>"; choice "<label>: <criteria[label]>"; score
    "<level>: <description of level>". A definition given as an object is written as one line of JSON; a label
    without a definition is written alone."""
    crit = criteria or {}
    if qtype == "noul":
        assert labels == ["no", "yes"], labels
        defs = {"no": crit.get("false"), "yes": crit.get("true")}
    elif qtype == "score":
        defs = {lab: crit[i] if i < len(crit) else None for i, lab in enumerate(labels)}
    else:
        defs = {lab: crit.get(lab) for lab in labels}
    text = {lab: d.strip() if isinstance(d, str) else json.dumps(d, ensure_ascii=False) if d else None
            for lab, d in defs.items()}
    return ["%s: %s" % (lab, text[lab]) if text[lab] else lab for lab in labels]


def typed_state(state):
    """Text states as they are; object and list states as indented JSON."""
    return state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)


def jevbench(raw, tier="hard"):
    """JevBench typed decisions (github.com/fstandhartinger/jevbench, MIT; public files only), test only, no depth.

    Options as in typed_options, in the benchmark's label order; states as in typed_state. The gold label is
    answer_index; the benchmark's family, question type and, for its `probability` items, the exact gold distribution
    go to meta."""
    out = []
    for r in (json.loads(l) for l in open(os.path.join(raw, "fstandhartinger__jevbench/datasets/public/%s.jsonl" % tier))
              if l.strip()):
        q, labels = r["question"], [str(x) for x in r["labels"]]
        options = typed_options(q["type"], labels, q.get("criteria"))
        state = typed_state(r["state"])
        prov = r.get("provenance") or {}
        out.append(item("jevbench_" + tier, "test", None, None, state, q["instructions"], options,
                        labels.index(str(r["expected"])), qtype=q["type"],
                        meta={"id": r["id"], "family": r["family"], "labels": labels,
                              "author_model": prov.get("author_model"), "surface_answer": prov.get("surface_answer"),
                              "gold_probs": prov.get("gold_probs")}))
    return out


KEV_SUITES = {"decision_v7": "v7/decision-v7", "transfer_v9": "v9/transfer-v9"}
KEV_META = ["id", "source", "variant", "group_id", "parent_id", "control_id", "none_key", "family"]   # kept from _meta


def kev(raw, suite, split, max_options=26, seed=0):
    """Kev's frozen suites (github.com/jaredpalmer/kev, Apache-2.0; each source keeps its own license): one item per
    question, sharing its record's state. Options as in typed_options (choice labels in the order of the question's
    criteria, noul no / yes, score levels 0..k-1); states as in typed_state; instructions that Kev writes as an object
    (its rendering variants, e.g. {"question": ..., "focus": ...}) as one line of JSON.

    The readout has one letter per option, so a question with more than max_options options (Banking77's 77 intents)
    keeps the gold option and max_options - 1 others drawn with a fixed seed, in their original order;
    meta.n_options_full keeps the original count. Kev's record fields in KEV_META (source, variant: clean / permuted /
    none_present / none_absent, the ids that pair variants and unknowable records with their controls) go to meta."""
    path = os.path.join(raw, "jaredpalmer__kev/evals", KEV_SUITES[suite], "%s.jsonl" % split)
    out = []
    for i, r in enumerate(json.loads(l) for l in open(path) if l.strip()):
        state = typed_state(r["state"])
        for name, q in r["questions"].items():
            if q["type"] == "noul":
                labels, gold = ["no", "yes"], "yes" if q["label"] else "no"
            elif q["type"] == "score":
                labels, gold = [str(j) for j in range(len(q["criteria"]))], str(q["label"])
            else:
                labels, gold = list(q["criteria"]), q["label"]
            assert gold in labels, (suite, split, i, name)
            n_full = len(labels)
            if n_full > max_options:
                rng = random.Random("%d/%s/%s/%d/%s" % (seed, suite, split, i, name))
                keep = set(rng.sample([l for l in labels if l != gold], max_options - 1)) | {gold}
                labels = [l for l in labels if l in keep]
            ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"], ensure_ascii=False)
            out.append(item("kev_" + suite, split, None, None, state, ins,
                            typed_options(q["type"], labels, q.get("criteria")), labels.index(gold), qtype=q["type"],
                            meta={"record": i, "question": name, "src": q.get("src"), "labels": labels,
                                  "n_options_full": n_full, **{k: r["_meta"].get(k) for k in KEV_META}}))
    return out


# ---------------------------------------------------------------- NLI (hard and soft labels)

def _nli_item(dataset, split, premise, hypothesis, answer_index=None, dist=None, meta=None):
    q = 'Given the text above, is this hypothesis entailed, neutral, or contradicted? Hypothesis: "%s"' % hypothesis.strip()
    return item(dataset, split, None, None, premise, q, NLI_OPTIONS, answer_index, dist, meta=meta)


def chaosnli(raw, sources):
    """ChaosNLI v1.0 (100 annotations per item), read from the earino/chaosnli preservation mirror."""
    out = []
    files = {"snli": "chaosNLI_snli.jsonl", "mnli": "chaosNLI_mnli_m.jsonl", "alphanli": "chaosNLI_alphanli.jsonl"}
    for src in sources:
        for r in (json.loads(l) for l in open(os.path.join(raw, "earino__chaosnli/raw", files[src]))):
            ex, dist = r["example"], r["label_dist"]
            meta = {"uid": r["uid"], "source": src, "entropy": r["entropy"], "old_labels": r.get("old_labels")}
            split = "test" if src == "mnli" else "train"
            if src == "alphanli":
                state = "Beginning: %s\nEnding: %s" % (ex["obs1"].strip(), ex["obs2"].strip())
                out.append(item("chaosnli_alphanli", split, None, None, state,
                                "Which explanation best connects the beginning and the ending?",
                                [ex["hyp1"].strip(), ex["hyp2"].strip()], dist=dist, meta=meta)
                           if ex["hyp1"].strip() != ex["hyp2"].strip() else None)
            else:
                out.append(_nli_item("chaosnli_" + src, split, ex["premise"], ex["hypothesis"], dist=dist, meta=meta))
    return [x for x in out if x is not None]
