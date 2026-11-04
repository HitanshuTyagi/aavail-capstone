# AAVAIL – 30-day Revenue Forecasting Service

**Business question:** *"Predict the revenue for the next 30 days, for a single country or for all countries combined."*
Finance can use the forecast to plan capacity and spend, and to spot countries that are slipping.

The solution is an end-to-end AI workflow: **ingestion → EDA → model comparison → API → logging → monitoring → tests → Docker**.

> **About the data.** The original AAVAIL invoice JSON files are not redistributable, so `scripts/generate_data.py`
> creates synthetic data with the *same schema* (monthly `invoices-YYYY-MM.json`, incl. the inconsistent
> `StreamID` / `TimesViewed` column names). To use real data, drop the files into `data/cs-train/` and
> `data/cs-production/` and run the pipeline below – no code changes needed. All numbers in this README come
> from the synthetic data.

## Quick start

```bash
pip install -r requirements.txt
python scripts/generate_data.py      # skip if you have the real data
python ingestion.py data/cs-train    # build cached daily time-series
python model.py                      # compare models, save best per country
python run-tests.py                  # ALL unit tests, one command
python app.py                        # API on http://localhost:8080
```

Docker:

```bash
docker build -t aavail-capstone .    # build also trains models and runs all tests
docker run -p 8080:8080 aavail-capstone
```

### API

```bash
# all countries combined
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
     -d '{"country":"all","year":2019,"month":8,"day":1}'
# one country
curl "localhost:8080/predict?country=germany&year=2019&month=7&day=15"
# retrain (mode "test" only touches test models/logs)
curl -X POST localhost:8080/train -H 'Content-Type: application/json' -d '{"mode":"test"}'
# logs / runtime summary
curl localhost:8080/logs/predict ; curl "localhost:8080/logs/predict?summary=1"
```

Valid countries: `all, united_kingdom, eire, germany, france, norway, spain, portugal, singapore, hong_kong, netherlands`.
Valid dates: any day in the available history (training + production data). Unknown country → `404`, bad/out-of-range date or missing field → `400`.

## Peer-review checklist → where to look

| Question | Answer / location |
|---|---|
| Unit tests for the API? | `unittests/ApiTests.py` (predict all + specific country, GET & POST, missing field, unknown country, bad date, logs endpoint) |
| Unit tests for the model? | `unittests/ModelTests.py` (training, model comparison, selection vs baseline, load, predict, error cases, feature shapes) |
| Unit tests for the logging? | `unittests/LoggerTests.py` (train log, predict log, append, test/prod separation) |
| All tests with one script, all pass? | `python run-tests.py` → **23 tests, OK** (exit code 0 only if all pass) |
| Performance monitoring? | `monitoring.py` → `reports/monitoring.json`: model vs baseline on production hold-out (RMSE/MAPE), KS-test data drift per feature, runtime stats from predict logs (also at `/logs/predict?summary=1`) |
| Tests isolated from production models/logs? | Every function takes `test=True`: models go to `models/test-*`, logs to `logs/*-test.log`; the API accepts `"test": true`; production data is never read in test mode. `LoggerTests.test_production_log_untouched` asserts it. |
| API works for one country and for all? | Yes – `country=all` or any country key (see above); covered by `ApiTests` |
| Ingestion as a function/script? | `ingestion.py` (`fetch_data`, `convert_to_ts`, `fetch_ts`, `engineer_features`); `python ingestion.py <dir>` rebuilds the cache |
| Multiple models compared? | Baseline + Ridge + Random Forest + Gradient Boosting, time-series cross-validation + grid search (`model.compare_models`). Results in `models/prod-model-comparison.csv` |
| EDA with visualizations? | `python scripts/eda.py` → `reports/figures/01…07_*.png` |
| Containerized in a working Docker image? | `Dockerfile` (python:3.11-slim, gunicorn). Build trains models and runs the test-suite, so a broken image fails the build |
| Visualization comparing model to baseline? | `python scripts/plot_model_comparison.py` → `08_model_vs_baseline_all.png`, `09_improvement_by_country.png`, `10_holdout_pred_vs_actual.png` |

## Approach

* **Target:** revenue in the 30 days *after* each date. **Features:** revenue over the previous 7/14/28/70 days, same window one year earlier, recent invoice/view activity, yearly seasonality (sin/cos of day-of-year).
* **Baseline:** "next 30 days = last 28 days × 30/28".
* **Candidates** learn the *correction to the baseline* (`estimators.ResidualModel`) – much more stable than predicting absolute revenue from ~600 daily rows.
* **Selection:** time-series CV (no look-ahead). The baseline competes too – if no ML model beats it for a country, the baseline is the one deployed (e.g. Germany).

## Results (synthetic data)

| | Baseline | Deployed model |
|---|---|---|
| CV RMSE, all countries | 2,197 | **1,391** (Ridge on residual) |
| Production hold-out RMSE (Aug–Oct 2019) | 2,435 | **546** |
| Production hold-out MAPE | 15.6 % | **2.9 %** |

For several small countries the baseline is not beaten (see `09_improvement_by_country.png`) – with so little signal per country, a simple rule is hard to beat, and the selection logic reflects that rather than hiding it.

## Monitoring notes

* The KS test flags drift in every feature on the hold-out. That is expected here: the production window (Aug–Nov) is compared with the whole training history, so it mixes a real trend with a seasonal shift. In production, compare like-for-like seasons, or alert on *model error* rising rather than on drift alone.
* Suggested follow-ups: retrain when `model_mape` on newly-matured data exceeds a threshold; schedule `ingestion.py` + `model.py` + `monitoring.py` (cron / Airflow).

## Project layout

```
app.py              Flask API                 ingestion.py   data loading + features
model.py            train / compare / predict estimators.py  baseline + residual estimators
logger.py           train/predict logging     monitoring.py  performance + drift + runtime
run-tests.py        runs all unit tests       unittests/     Api / Model / Logger tests
scripts/            generate_data, eda, plot_model_comparison
data/               cs-train, cs-production   models/ logs/ reports/figures/
Dockerfile  requirements.txt
```
