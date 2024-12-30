from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
from utils import predict_traffic

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = "app/model/vehicle_traffic_prediction_model.pkl"
SCALER_PATH = "app/model/vehicle_traffic_scaler_total.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load future forecast data
FUTURE_FORECAST_PATH = "dataset/future_traffic_forecast.parquet"
future_traffic = pd.read_parquet(FUTURE_FORECAST_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    site = data.get("site")
    date = pd.to_datetime(data.get("date")).date()
    time_of_day = data.get("time_of_day")

    prediction = predict_traffic(model, scaler, future_traffic, site, date, time_of_day)

    if prediction:
        return jsonify({
            "total": round(prediction["total"], 2),
            "northbound": round(prediction["northbound"], 2),
            "southbound": round(prediction["southbound"], 2),
            "eastbound": round(prediction["eastbound"], 2),
            "westbound": round(prediction["westbound"], 2)
        })
    else:
        return jsonify({"error": "No forecasted data available for the selected input."}), 404

if __name__ == "__main__":
    app.run(debug=True)
