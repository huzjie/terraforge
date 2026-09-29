"""Metric helpers: confusion stats, ECE, ranking metrics."""
import math


def confusion(preds, targets):
    tp = fp = tn = fn = 0
    for p, t in zip(preds, targets):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and t:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision(conf):
    return conf["tp"] / (conf["tp"] + conf["fp"]) if (conf["tp"] + conf["fp"]) else 0.0


def recall(conf):
    return conf["tp"] / (conf["tp"] + conf["fn"]) if (conf["tp"] + conf["fn"]) else 0.0


def f1(conf):
    p, r = precision(conf), recall(conf)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy(preds, targets):
    if not targets:
        return 0.0
    return sum(1 for p, t in zip(preds, targets) if p == t) / len(targets)


def expected_calibration_error(confs, corrects, n_bins=10):
    if not confs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confs, corrects):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, ok))
    ece = 0.0
    total = len(confs)
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / total) * abs(acc - conf)
    return ece


def mean_reciprocal_rank(rankings):
    """rankings: list of lists of true-item ranks (0-indexed)."""
    if not rankings:
        return 0.0
    return sum(1.0 / (r + 1) for r in rankings) / len(rankings)
