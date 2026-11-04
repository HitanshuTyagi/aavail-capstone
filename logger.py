"""
Logging for train / predict calls.

Logs are written to CSV files. `test=True` redirects every write to a
*separate* test log file so unit tests never touch production logs.
"""

import os
import csv
import time
import uuid
from datetime import date

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

TRAIN_HEADER = ['unique_id', 'timestamp', 'x_shape', 'eval_test',
                'model_version', 'model_version_note', 'runtime']
PREDICT_HEADER = ['unique_id', 'timestamp', 'country', 'y_pred',
                  'y_proba', 'query', 'model_version', 'runtime']


def _log_path(kind, test):
    """kind in {'train','predict'}; test logs are fully separate files."""
    if test:
        return os.path.join(LOG_DIR, f"{kind}-test.log")
    return os.path.join(LOG_DIR, f"{kind}-{date.today().year}-{date.today().month}.log")


def _write(path, header, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def update_train_log(data_shape, eval_test, runtime, model_version,
                     model_version_note, test=False):
    row = [str(uuid.uuid4()), time.time(), data_shape, eval_test,
           model_version, model_version_note, runtime]
    _write(_log_path('train', test), TRAIN_HEADER, row)


def update_predict_log(country, y_pred, y_proba, query, runtime,
                       model_version, test=False):
    row = [str(uuid.uuid4()), time.time(), country, y_pred, y_proba,
           query, model_version, runtime]
    _write(_log_path('predict', test), PREDICT_HEADER, row)
