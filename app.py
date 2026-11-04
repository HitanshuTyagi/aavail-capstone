"""
Flask API for the AAVAIL revenue forecaster.

  GET  /                 -> health / info
  POST /train            -> retrain models  {"mode": "test" | "prod"}
  POST /predict          -> {"country": "all"|"germany"|..., "year": 2019, "month": 8, "day": 1}
  GET  /predict?country=all&year=2019&month=8&day=1
  GET  /logs/<train|predict>  -> tail of the log (monitoring)
"""

import os
import argparse
from flask import Flask, jsonify, request, abort

from model import model_train, model_predict, MODEL_VERSION
from logger import LOG_DIR
from monitoring import service_report

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data", "cs-train"))

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"service": "AAVAIL 30-day revenue forecaster",
                    "model_version": MODEL_VERSION,
                    "endpoints": ["/predict", "/train", "/logs/<train|predict>"]})


def _bool(v):
    return str(v).lower() in ("1", "true", "yes", "test")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    payload = request.get_json(silent=True) if request.method == "POST" else request.args
    payload = payload or {}
    try:
        country = str(payload.get("country", "all")).lower().replace(" ", "_")
        year, month, day = payload["year"], payload["month"], payload["day"]
    except KeyError as e:
        return jsonify({"error": f"missing field {e}"}), 400
    test = _bool(payload.get("test", False))
    try:
        return jsonify(model_predict(country, year, month, day, test=test))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    test = _bool(payload.get("mode", "prod") == "test")
    table = model_train(DATA_DIR, test=test)
    return jsonify({"trained": True, "mode": "test" if test else "prod",
                    "comparison": table.to_dict(orient="records")})


@app.route("/logs/<kind>")
def logs(kind):
    if kind not in ("train", "predict"):
        abort(404)
    test = _bool(request.args.get("test", False))
    if kind == "predict" and request.args.get("summary"):
        return jsonify(service_report(test=test))
    name = f"{kind}-test.log" if test else None
    files = sorted(f for f in os.listdir(LOG_DIR)
                   if f.startswith(kind + "-") and (f == name if test else not f.endswith("-test.log")))
    if not files:
        return jsonify({"error": "no log file"}), 404
    with open(os.path.join(LOG_DIR, files[-1])) as f:
        return jsonify({"file": files[-1], "tail": f.read().splitlines()[-50:]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true")
    args = ap.parse_args()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=args.debug)
