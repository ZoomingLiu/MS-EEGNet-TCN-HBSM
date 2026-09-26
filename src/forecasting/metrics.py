"""Endpoint metrics with grouped ties; no inflated accuracy-only summary."""

import numpy as np


def probability_metrics(labels, probabilities, baseline_probability=None):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Metrics require aligned, nonempty observed binary labels and probabilities")
    y = y.astype(int)
    if baseline_probability is not None and (not np.isfinite(baseline_probability) or not 0 <= baseline_probability <= 1):
        raise ValueError("Baseline must be a finite probability in [0, 1]")
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), len(ps) - 1]
    tp = np.cumsum(ys)[ends].astype(float)
    fp = (ends + 1) - tp
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    ap, auc = None, None
    if n_pos:
        recalls = np.r_[0, tp / n_pos]
        ap = float(np.sum(np.diff(recalls) * tp / (tp + fp)))
    if n_pos and n_neg:
        tpr, fpr = np.r_[0, tp / n_pos], np.r_[0, fp / n_neg]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    result = {"n": len(y), "positive_count": n_pos, "prevalence": float(y.mean()), "average_precision": ap,
              "roc_auc": auc, "brier": float(np.mean((p - y) ** 2)), "accuracy_at_0_5": float(np.mean((p >= .5) == y))}
    if baseline_probability is not None:
        baseline = float(np.mean((float(baseline_probability) - y) ** 2))
        result.update(baseline_brier=baseline, brier_skill_score=1 - result["brier"] / baseline if baseline > 0 else None)
    return result
