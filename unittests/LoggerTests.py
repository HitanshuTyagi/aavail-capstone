"""Logging unit tests. All writes use test=True => logs/*-test.log, never the
production log files."""

import os
import sys
import csv
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import update_train_log, update_predict_log, LOG_DIR, _log_path

TRAIN = os.path.join(LOG_DIR, "train-test.log")
PREDICT = os.path.join(LOG_DIR, "predict-test.log")


def _rows(path):
    with open(path) as f:
        return list(csv.reader(f, delimiter=",", quotechar="|"))


class LoggerTest(unittest.TestCase):

    def setUp(self):
        for p in (TRAIN, PREDICT):
            if os.path.exists(p):
                os.remove(p)

    def test_test_logs_are_separate_from_prod(self):
        self.assertTrue(_log_path("train", True).endswith("train-test.log"))
        self.assertNotEqual(_log_path("train", True), _log_path("train", False))
        self.assertNotEqual(_log_path("predict", True), _log_path("predict", False))

    def test_train_log(self):
        update_train_log("(100, 6)", {"rmse": 1.0}, "0:00:01", 0.1, "unit test", test=True)
        rows = _rows(TRAIN)
        self.assertEqual(len(rows), 2)                       # header + 1
        self.assertEqual(rows[0][0], "unique_id")
        self.assertEqual(rows[1][4], "0.1")

    def test_predict_log(self):
        update_predict_log("all", 123.4, None, "2019-6-1", "0.01s", 0.1, test=True)
        rows = _rows(PREDICT)
        self.assertEqual(rows[1][2], "all")
        self.assertEqual(rows[1][3], "123.4")

    def test_log_appends(self):
        for _ in range(3):
            update_predict_log("all", 1.0, None, "q", "0.01s", 0.1, test=True)
        self.assertEqual(len(_rows(PREDICT)), 4)

    def test_production_log_untouched(self):
        prod = _log_path("predict", False)
        before = os.path.getsize(prod) if os.path.exists(prod) else None
        update_predict_log("all", 1.0, None, "q", "0.01s", 0.1, test=True)
        after = os.path.getsize(prod) if os.path.exists(prod) else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
