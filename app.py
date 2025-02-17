from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your LSTM model
model = tf.keras.models.load_model("BitCoin_PRE_AI.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "input" not in data:
        return jsonify({"error": "Invalid input"}), 400

    # Convert input to numpy array and reshape
    input_data = np.array(data["input"]).reshape(1, -1, 1)

    # Make prediction
    prediction = model.predict(input_data).tolist()

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
