"""Shortcut probes: can a cheap learner predict the label without doing the task?

Two probes per family, both trained on the train split and scored on the test split:

  1. bag of words: hashed unigrams + bigrams of the text, logistic regression (numpy).
  2. heuristics: each numeric feature in the family's HEURISTICS, best threshold rule
     (and best value-equality rule) picked on train.

Both should score close to the majority class. A probe that clearly beats majority
means the data leaks the label through its surface, and the generator must change.
"""
import re
from collections import Counter

import numpy as np

DIM = 1 << 15


def _tokens(text):
    return re.findall(r"[A-Za-z']+|\d+|\$", text.lower())


def featurize(items):
    X = np.zeros((len(items), DIM), dtype=np.float32)
    for i, it in enumerate(items):
        toks = _tokens(it["state"] + " " + it["question"])
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            X[i, hash(g) % DIM] += 1.0
        X[i] = np.log1p(X[i])
    return X


def train_logreg(X, y, n_classes, epochs=60, lr=0.5, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    n = len(X)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), max(1, n // 64)):
            z = X[idx] @ W + b
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(idx)), y[idx]] -= 1.0
            W -= lr * (X[idx].T @ p / len(idx) + l2 * W)
            b -= lr * p.mean(axis=0)
    return W, b


def bow_probe(train, test):
    n_classes = len(train[0]["options"])
    Xtr, Xte = featurize(train), featurize(test)
    ytr = np.array([it["answer_index"] for it in train])
    yte = np.array([it["answer_index"] for it in test])
    W, b = train_logreg(Xtr, ytr, n_classes)
    pred = (Xte @ W + b).argmax(axis=1)
    return float((pred == yte).mean())


def heuristic_probe(train, test, heuristics):
    """For each heuristic, the best threshold rule (or equality rule) on train, scored on test."""
    out = {}
    for name, fn in heuristics.items():
        ftr = np.array([fn(it) for it in train], dtype=float)
        fte = np.array([fn(it) for it in test], dtype=float)
        ytr = np.array([it["answer_index"] for it in train])
        yte = np.array([it["answer_index"] for it in test])
        best, best_rule = 0.0, None
        for t in np.unique(ftr):
            for kind in ("le", "eq"):
                m = (ftr <= t) if kind == "le" else (ftr == t)
                if m.sum() == 0 or (~m).sum() == 0:
                    continue
                c_in = Counter(ytr[m]).most_common(1)[0][0]
                c_out = Counter(ytr[~m]).most_common(1)[0][0]
                acc = (np.where(m, c_in, c_out) == ytr).mean()
                if acc > best:
                    best, best_rule = acc, (kind, float(t), int(c_in), int(c_out))
        if best_rule is None:
            out[name] = None
            continue
        kind, t, c_in, c_out = best_rule
        m = (fte <= t) if kind == "le" else (fte == t)
        out[name] = float((np.where(m, c_in, c_out) == yte).mean())
    return out


def majority(test):
    c = Counter(it["answer_index"] for it in test)
    return c.most_common(1)[0][1] / len(test)


def run(train, test, heuristics, margin=0.05):
    """Returns a report dict; 'flags' lists probes that beat majority by more than `margin`."""
    maj = majority(test)
    bow = bow_probe(train, test)
    heur = heuristic_probe(train, test, heuristics)
    flags = []
    if bow > maj + margin:
        flags.append(f"bow={bow:.3f}")
    for name, acc in heur.items():
        if acc is not None and acc > maj + margin:
            flags.append(f"{name}={acc:.3f}")
    return {"majority": round(maj, 3), "bag_of_words": round(bow, 3),
            "heuristics": {k: (None if v is None else round(v, 3)) for k, v in heur.items()},
            "flags": flags}
