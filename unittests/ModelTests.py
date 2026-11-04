"""Model unit tests (isolated: test=True => models/test-* only)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion import fetch_ts, engineer_features
from model import model_train, model_predict, model_load, compare_models, MODEL_DIR

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cs-train")


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.table = model_train(DATA_DIR, test=True)

    def test_train_saves_test_model_only(self):
        files = os.listdir(MODEL_DIR)
        self.assertTrue(any(f.startswith("test-all") for f in files))

    def test_multiple_models_compared(self):
        models = set(self.table["model"])
        self.assertTrue({"baseline", "ridge", "random_forest", "gradient_boosting"} <= models)

    def test_exactly_one_model_selected(self):
        self.assertTrue((self.table.groupby("country")["selected"].sum() == 1).all())

    def test_selected_beats_baseline(self):
        t = self.table[self.table["country"] == "all"].set_index("model")
        best = t[t["selected"]]["rmse"].min()
        self.assertLess(best, t.loc["baseline", "rmse"])

    def test_load(self):
        art = model_load("all", test=True)
        self.assertIn("model", art)

    def test_load_missing_country(self):
        with self.assertRaises(FileNotFoundError):
            model_load("atlantis", test=True)

    def test_predict_value(self):
        r = model_predict("all", 2019, 6, 1, test=True)
        self.assertGreater(r["y_pred"], 0)
        self.assertEqual(r["country"], "all")

    def test_predict_out_of_range(self):
        with self.assertRaises(ValueError):
            model_predict("all", 1990, 1, 1, test=True)

    def test_predict_invalid_date(self):
        with self.assertRaises(ValueError):
            model_predict("all", 2019, 13, 40, test=True)

    def test_feature_engineering_shapes(self):
        ts = fetch_ts(DATA_DIR)["all"]
        X, y, dates = engineer_features(ts, training=True)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(dates))
        self.assertFalse(np.isnan(X.values).any())
        Xp, _, dp = engineer_features(ts, training=False)
        self.assertEqual(len(Xp), len(ts))


if __name__ == "__main__":
    unittest.main()
