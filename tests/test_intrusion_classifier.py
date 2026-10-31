import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from intrusion_classifier import (
    build_pipeline,
    select_features_by_importance,
    split_data,
)


class IntrusionClassifierTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            {
                "bytes": np.arange(100),
                "protocol": ["tcp", "udp"] * 50,
            }
        )
        self.target = pd.Series([0, 1] * 50)

    def test_split_data_creates_disjoint_stratified_sets(self):
        splits = split_data(self.features, self.target)

        index_sets = [
            set(splits.X_train.index),
            set(splits.X_validation.index),
            set(splits.X_test.index),
        ]
        self.assertEqual([len(index) for index in index_sets], [64, 16, 20])
        self.assertFalse(index_sets[0] & index_sets[1])
        self.assertFalse(index_sets[0] & index_sets[2])
        self.assertFalse(index_sets[1] & index_sets[2])
        self.assertEqual(splits.y_test.value_counts().to_dict(), {0: 10, 1: 10})

    def test_pipeline_handles_categories_unseen_during_training(self):
        pipeline = build_pipeline(
            RandomForestClassifier(n_estimators=5, random_state=42),
            self.features,
        )
        pipeline.fit(self.features, self.target)

        predictions = pipeline.predict(
            pd.DataFrame({"bytes": [101], "protocol": ["icmp"]})
        )

        self.assertEqual(predictions.shape, (1,))

    def test_feature_selection_includes_threshold_crossing_feature(self):
        importances = pd.Series({"first": 0.6, "second": 0.3, "third": 0.1})

        selected = select_features_by_importance(importances, threshold=0.95)

        self.assertEqual(selected.index.tolist(), ["first", "second", "third"])
        self.assertGreaterEqual(selected.sum(), 0.95)


if __name__ == "__main__":
    unittest.main()
