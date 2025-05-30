# example: app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["sequence"]
        if len(data) != 60:
            return jsonify({"error": "Input must be 60 float values."}), 400

        input_seq = np.array(data).reshape(-1, 1)
        scaled = scaler.transform(input_seq).reshape(1, 60, 1)
        prediction = model.predict(scaled)
        result = scaler.inverse_transform(prediction)[0][0]

        return jsonify({"predicted_price": float(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
