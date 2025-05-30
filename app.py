# app.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["sequence"]  # Expecting 60 values (float)

    if len(data) != 60:
        return jsonify({"error": "Input sequence must be 60 time steps long."}), 400

    input_seq = np.array(data).reshape(1, 60, 1)
    scaled_input = scaler.transform(np.array(data).reshape(-1, 1)).reshape(1, 60, 1)

    prediction = model.predict(scaled_input)
    pred_price = scaler.inverse_transform(prediction)[0][0]

    return jsonify({"predicted_price": float(pred_price)})

# Test route
@app.route("/")
def home():
    return "LSTM model is deployed and running!"

