from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo
model = joblib.load("iris_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "API del modelo Iris funcionando correctamente"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "input" not in data:
        return jsonify({"error": "Debe enviar un campo llamado input"}), 400

    input_data = np.array(data["input"]).reshape(1, -1)

    prediction = model.predict(input_data)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)