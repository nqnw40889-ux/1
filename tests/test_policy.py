import unittest

import numpy as np
import pandas as pd

from evaluation import measure
from policy import cost_of, select_thresholds, threshold_curve


class PolicyTests(unittest.TestCase):
    def test_exhaustive_thresholds_group_ties_and_include_zero_alerts(self):
        labels = np.array([1, 0, 0, 1, 1, 0])
        scores = np.array([.2, .8, .1, .8, .4, .8])
        curve = threshold_curve(labels, scores)
        self.assertEqual(len(curve), 5)
        self.assertEqual(int(curve.iloc[0].TP + curve.iloc[0].FP), 0)
        for row in curve.itertuples():
            m = measure(labels, scores, row.Threshold)
            self.assertEqual((m['TP'], m['FP'], m['FN'], m['TN']),
                             (row.TP, row.FP, row.FN, row.TN))

    def test_deployed_policy_minimizes_validation_cost(self):
        import joblib
        model = joblib.load('mo_hinh_demo.joblib')
        validation = pd.read_csv('reports/predictions_validation.csv', float_precision='round_trip')
        curve = threshold_curve(validation.actual, validation.probability)
        miss_cost = model['metadata']['miss_cost']
        thresholds = select_thresholds(curve, miss_cost)
        self.assertEqual(thresholds, model['metadata']['policies'])
        self.assertEqual(thresholds['Cost'], model['threshold'])
        metrics = measure(validation.actual, validation.probability, model['threshold'])
        self.assertEqual(cost_of(metrics, miss_cost), int(cost_of(curve, miss_cost).min()))
