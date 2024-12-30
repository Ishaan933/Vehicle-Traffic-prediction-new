from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model, scaler, and forecast data
MODEL_PATH = "app/model/vehicle_traffic_prediction_model.pkl"
SCALER_PATH = "app/model/vehicle_traffic_scaler_total.pkl"
FUTURE_FORECAST_PATH = "dataset/future_traffic_forecast.parquet"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
future_traffic = pd.read_parquet(FUTURE_FORECAST_PATH)

# Prediction function
def predict_traffic(model, scaler, future_traffic, site, date, time_of_day):
    time_of_day_features = {
        'Morning': [1, 0, 0, 0],
        'Afternoon': [0, 1, 0, 0],
        'Evening': [0, 0, 1, 0],
        'Night': [0, 0, 0, 1]
    }

    future_traffic['ds'] = pd.to_datetime(future_traffic['ds']).dt.date
    future_traffic['Site'] = future_traffic['Site'].str.strip().str.title()  # Change to title case
    site = site.strip().title()  # Change to title case

    future_row = future_traffic[(future_traffic['Site'] == site) & (future_traffic['ds'] == date)]

    if future_row.empty:
        return None

    northbound = future_row['Northbound'].values[0]
    southbound = future_row['Southbound'].values[0]
    eastbound = future_row['Eastbound'].values[0]
    westbound = future_row['Westbound'].values[0]

    time_of_day_values = time_of_day_features[time_of_day]

    future_input = pd.DataFrame({
        'Northbound': [northbound],
        'Southbound': [southbound],
        'Eastbound': [eastbound],
        'Westbound': [westbound],
        'TimeOfDay_Morning': [time_of_day_values[0]],
        'TimeOfDay_Afternoon': [time_of_day_values[1]],
        'TimeOfDay_Evening': [time_of_day_values[2]],
        'TimeOfDay_Night': [time_of_day_values[3]],
        **{f'Site_{s}': [int(s == site)] for s in future_traffic['Site'].unique()}
    })

    # Ensure input columns match training columns
    for col in model.feature_names_in_:
        if col not in future_input.columns:
            future_input[col] = 0
    future_input = future_input[model.feature_names_in_]

    predicted_scaled = model.predict(future_input)[0]
    predicted_total = scaler.inverse_transform([[predicted_scaled]])[0][0]

    return {
        'total': predicted_total,
        'northbound': northbound,
        'southbound': southbound,
        'eastbound': eastbound,
        'westbound': westbound
    }

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
 
if __name__ == '__main__':
    # Use port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
