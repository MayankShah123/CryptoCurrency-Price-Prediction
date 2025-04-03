from flask import Flask, request, jsonify, render_template
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")  # Set templates folder
CORS(app)  # Enable CORS

# Load trained model (Ensure the model file exists)
model = load_model("./model/trained_model.h5")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Load the frontend

@app.route("/predict", methods=["GET"])
def predict():
    crypto_symbol = request.args.get("symbol", "").strip()

    if not crypto_symbol:
        return jsonify({"error": "Missing 'symbol' parameter"}), 400

    # Download latest data
    try:
        data = yf.download(crypto_symbol, period="30d", interval="1d")  # Fetch last 30 days
        if data.empty:
            return jsonify({"error": "Invalid cryptocurrency symbol or no data available"}), 400

        actual_today = float(data['Close'].iloc[-1])
        actual_yesterday = float(data['Close'].iloc[-2]) if len(data) > 1 else actual_today

        # Prepare data for prediction
        look_back = 30
        close_prices = data['Close'].values[-look_back:].reshape(-1, 1)

        if len(close_prices) < look_back:
            return jsonify({"error": f"Not enough data. Need {look_back}, but got {len(close_prices)}."})

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_scaled = scaler.fit_transform(close_prices)

        # Reshape for LSTM model
        close_prices_scaled = close_prices_scaled.reshape(1, look_back, 1)

        # Predict next day's price
        predicted_next_day_scaled = model.predict(close_prices_scaled)
        predicted_next_day = scaler.inverse_transform(predicted_next_day_scaled)

        # Determine trend
        price_trend = "UP" if predicted_next_day[0][0] > actual_today else "DOWN"

        return jsonify({
            "symbol": crypto_symbol,
            "actual_today": actual_today,
            "actual_yesterday": actual_yesterday,
            "trend": price_trend
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
