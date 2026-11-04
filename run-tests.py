#!/usr/bin/env python
"""Run ALL unit tests (API, model, logger) with one command:

    python run-tests.py

Exit code is 0 only if every test passes.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    # test order: logger -> model (trains the isolated test model) -> api
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for mod in ("unittests.LoggerTests", "unittests.ModelTests", "unittests.ApiTests"):
        suite.addTests(loader.loadTestsFromName(mod))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
