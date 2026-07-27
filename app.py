import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

import warnings
warnings.filterwarnings("ignore")

print("[STARTUP] Loading TensorFlow...")
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    print(f"[STARTUP] TensorFlow {tf.__version__} loaded OK")
except Exception as e:
    print(f"[STARTUP] TensorFlow not available: {e}")

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import io, traceback, threading, uuid
from datetime import datetime, timedelta

from utils.lstm_model   import generate_sample_data, train_and_predict
from utils.data_sources import (get_weather_data, get_covid_data,
                                get_sales_data,   get_energy_data)

# Create downloads folder automatically
DOWNLOADS_FOLDER = os.path.join(os.path.dirname(__file__), "downloaded_data")
os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
print(f"[STARTUP] Downloads folder: {DOWNLOADS_FOLDER}")
from utils.data_sources import (get_weather_data, get_covid_data,
                                get_sales_data,   get_energy_data)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

JOBS = {}
JOBS_LOCK = threading.Lock()


def run_training_job(job_id, payload):
    def update(msg, pct=None):
        with JOBS_LOCK:
            JOBS[job_id]["msg"] = msg
            if pct is not None:
                JOBS[job_id]["pct"] = pct
        print(f"  [JOB] {msg}")
    try:
        update("Preprocessing data...", 10)
        values       = payload["values"]
        dates        = payload.get("dates", [])
        lookback     = int(payload.get("lookback",     30))
        epochs       = int(payload.get("epochs",       30))
        units        = int(payload.get("units",        64))
        dropout      = float(payload.get("dropout",    0.2))
        future_steps = int(payload.get("future_steps", 30))
        result = train_and_predict(
            values=values, lookback=lookback, epochs=epochs,
            units=units, dropout=dropout, future_steps=future_steps,
            progress_cb=lambda msg, pct: update(msg, pct),
        )
        if dates:
            try:
                last_date = pd.Timestamp(str(dates[-1]))
                result["future_dates"] = [
                    (last_date + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d")
                    for i in range(future_steps)
                ]
            except Exception:
                result["future_dates"] = list(range(future_steps))
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["pct"]    = 100
            JOBS[job_id]["msg"]    = "Complete!"
        print(f"  [JOB] DONE — mae={result.get('metrics',{}).get('mae')}")
    except Exception as e:
        traceback.print_exc()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"]  = str(e)
            JOBS[job_id]["msg"]    = f"Error: {e}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample-data")
def sample_data():
    try:
        data = generate_sample_data(
            request.args.get("type", "stock"),
            int(request.args.get("points", 300))
        )
        return jsonify({"success": True, **data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── New category datasets ─────────────────────────────────────────────────────
@app.route("/api/category-data", methods=["POST"])
def category_data():
    try:
        body     = request.get_json(force=True, silent=True) or {}
        category = body.get("category", "weather")
        option   = body.get("option",   "")

        if category == "weather":
            data = get_weather_data(city=option or "Hyderabad")
        elif category == "covid":
            data = get_covid_data(country=option or "India")
        elif category == "sales":
            data = get_sales_data(company=option or "E-Commerce Store")
        elif category == "energy":
            data = get_energy_data(city=option or "Delhi")
        else:
            return jsonify({"success": False, "error": "Unknown category"}), 400

        # Save CSV automatically to downloaded_data folder
        name     = (option or category).replace(" ", "_").replace("/", "_")
        filename = f"{category}_{name}.csv"
        filepath = os.path.join(DOWNLOADS_FOLDER, filename)
        df_save  = pd.DataFrame({"Date": data["dates"], "Value": data["values"]})
        df_save.to_csv(filepath, index=False)
        print(f"[SAVE] CSV saved: {filepath}")
        data["saved_file"] = filename

        return jsonify({"success": True, **data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def train():
    try:
        body     = request.get_json(force=True, silent=True) or {}
        values   = body.get("values", [])
        lookback = int(body.get("lookback", 30))
        if len(values) < lookback * 2:
            return jsonify({"success": False,
                "error": f"Need at least {lookback*2} data points (got {len(values)})."}), 400
        job_id = str(uuid.uuid4())
        with JOBS_LOCK:
            JOBS[job_id] = {"status":"running","msg":"Starting...","pct":0,"result":None,"error":None}
        t = threading.Thread(target=run_training_job, args=(job_id, body), daemon=True)
        t.start()
        print(f"[TRAIN] Job {job_id[:8]} started")
        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/status/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success":True,"status":job["status"],"msg":job["msg"],
                    "pct":job["pct"],"error":job["error"]})


@app.route("/api/result/<job_id>")
def job_result(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"success": False, "error": "Not finished yet"}), 400
    return jsonify({"success": True, **job["result"]})


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        f  = request.files["file"]
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return jsonify({"success": False, "error": "No numeric columns found"}), 400
        val_col = num_cols[0]
        values  = df[val_col].dropna().tolist()
        date_col = None
        for col in df.columns:
            try:
                pd.to_datetime(df[col]); date_col = col; break
            except Exception:
                continue
        dates = (pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d").tolist()
                 if date_col else list(range(len(values))))
        return jsonify({"success":True,"values":values,"dates":dates,
                        "label":val_col,"columns":df.columns.tolist(),"rows":len(df)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/download-stock", methods=["POST"])
def download_stock():
    try:
        body   = request.get_json(force=True, silent=True) or {}
        symbol = body.get("symbol", "RELIANCE.NS")
        try:
            import yfinance as yf
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
            import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=365 * 4)
        ticker = yf.Ticker(symbol)
        df     = ticker.history(start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"))
        if df.empty:
            return jsonify({"success": False,
                "error": f"No data found for {symbol}. Check internet connection."}), 400
        df = df[["Close"]].reset_index()
        df.columns = ["Date", "Close"]
        df["Date"]  = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df["Close"] = df["Close"].round(2)
        df = df.dropna()
        # Save CSV automatically to downloaded_data folder
        filename = f"stock_{symbol.replace('.', '_')}.csv"
        filepath = os.path.join(DOWNLOADS_FOLDER, filename)
        df.to_csv(filepath, index=False)
        print(f"[SAVE] CSV saved: {filepath}")

        return jsonify({
            "success": True, "symbol": symbol,
            "dates":   df["Date"].tolist(),
            "values":  df["Close"].tolist(),
            "label":   f"{symbol} Close Price (Rs.)",
            "rows":    len(df),
            "min":     round(float(df["Close"].min()), 2),
            "max":     round(float(df["Close"].max()), 2),
            "from":    df["Date"].iloc[0],
            "to":      df["Date"].iloc[-1],
            "saved_file": filename,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  LSTM Predictor — http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000, threaded=True)