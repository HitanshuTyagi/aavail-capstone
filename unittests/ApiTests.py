"""API unit tests. Uses the Flask test client and test=True so production
models/logs are never read or written."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from logger import LOG_DIR

TEST_PRED_LOG = os.path.join(LOG_DIR, "predict-test.log")


class ApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        r = cls.client.post("/train", json={"mode": "test"})
        assert r.status_code == 200, r.data

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("model_version", r.get_json())

    def test_predict_all_countries_combined(self):
        r = self.client.post("/predict", json={"country": "all", "year": 2019,
                                               "month": 6, "day": 1, "test": True})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["country"], "all")
        self.assertIsInstance(body["y_pred"], float)
        self.assertGreater(body["y_pred"], 0)

    def test_predict_get_query_string(self):
        r = self.client.get("/predict?country=all&year=2019&month=6&day=1&test=true")
        self.assertEqual(r.status_code, 200)

    def test_predict_specific_country(self):
        """Predictions for a single country (test mode trains 'all' + 'germany')."""
        r = self.client.post("/predict", json={"country": "germany", "year": 2019,
                                               "month": 6, "day": 1, "test": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], "germany")

    def test_missing_field(self):
        r = self.client.post("/predict", json={"country": "all", "year": 2019, "test": True})
        self.assertEqual(r.status_code, 400)

    def test_unknown_country(self):
        r = self.client.post("/predict", json={"country": "atlantis", "year": 2019,
                                               "month": 6, "day": 1, "test": True})
        self.assertEqual(r.status_code, 404)

    def test_bad_date(self):
        r = self.client.post("/predict", json={"country": "all", "year": 1999,
                                               "month": 1, "day": 1, "test": True})
        self.assertEqual(r.status_code, 400)

    def test_logs_endpoint(self):
        self.client.post("/predict", json={"country": "all", "year": 2019,
                                           "month": 6, "day": 1, "test": True})
        r = self.client.get("/logs/predict?test=true")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/logs/bogus").status_code, 404)


if __name__ == "__main__":
    unittest.main()
