"""Binary replay-detection metrics for anomaly scores."""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(anomaly_scores, labels, normal_label=1):
    """Compute EER/AUC where larger scores indicate greater abnormality."""
    scores = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    normal = labels == int(normal_label)
    anomaly = ~normal
    if not normal.any() or not anomaly.any():
        raise ValueError("evaluation requires normal and anomalous samples")
    order = np.argsort(scores, kind="mergesort")
    false_alarm = np.cumsum(anomaly[order]) / float(np.sum(anomaly))
    miss = 1.0 - np.cumsum(normal[order]) / float(np.sum(normal))
    index = int(np.argmin(np.abs(false_alarm - miss)))
    threshold = float(scores[order[index]])
    eer = float((false_alarm[index] + miss[index]) / 2.0)
    predicted_anomaly = scores > threshold
    return {
        "eer": eer,
        "threshold": threshold,
        "auc": float(roc_auc_score(anomaly.astype(np.int8), scores)),
        "false_acceptance_rate": float(np.mean(
            ~predicted_anomaly[anomaly])),
        "false_rejection_rate": float(np.mean(
            predicted_anomaly[normal])),
        "normal_score_mean": float(np.mean(scores[normal])),
        "normal_score_std": float(np.std(scores[normal])),
        "anomaly_score_mean": float(np.mean(scores[anomaly])),
        "anomaly_score_std": float(np.std(scores[anomaly])),
    }
