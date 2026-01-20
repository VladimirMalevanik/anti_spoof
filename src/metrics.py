from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    EER: точка, где FPR ~= FNR (1-TPR).
    y_true: 0/1
    y_score: P(spoof)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)
