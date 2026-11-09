"""Decision thresholds chosen from labeled validation scores."""
import numpy as np
import pandas as pd

DEFAULT_MISS_COST = 5
FALSE_ALARM_COST = 1


def threshold_curve(labels, scores):
    """Evaluate each distinct prediction set, including zero alerts."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    labels, scores = labels[order], scores[order]
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    tp = np.r_[0, labels.cumsum()[ends]]
    fp = np.r_[0, (1 - labels).cumsum()[ends]]
    fn = labels.sum() - tp
    tn = (1 - labels).sum() - fp
    return pd.DataFrame({
        'Threshold': np.r_[np.nextafter(scores[0], np.inf), scores[ends]],
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'F1': np.divide(2 * tp, 2 * tp + fp + fn),
    })


def select_thresholds(curve, miss_cost=DEFAULT_MISS_COST):
    # The curve is ordered by descending threshold. Ties favor fewer alerts.
    costs = miss_cost * curve['FN'] + FALSE_ALARM_COST * curve['FP']
    return {
        'Cost': float(curve.loc[costs.idxmin(), 'Threshold']),
        'F1': float(curve.loc[curve['F1'].idxmax(), 'Threshold']),
    }


def cost_of(metrics, miss_cost=DEFAULT_MISS_COST):
    return miss_cost * metrics['FN'] + FALSE_ALARM_COST * metrics['FP']
