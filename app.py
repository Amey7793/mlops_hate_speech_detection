"""
Flask REST API for hate speech detection using the registered RF model from MLflow.
"""
import os
import pickle
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from src.predict import predict
from src import config

app = Flask(__name__)

if os.getenv("USE_PKL", "false").lower() == "true":
    with open(config.VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(config.RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
else:
    client = mlflow.MlflowClient()
    rf_version = client.get_registered_model("rf_model").latest_versions[0]
    rf_model = mlflow.sklearn.load_model("models:/rf_model/latest")
    vectorizer_path = mlflow.artifacts.download_artifacts(
        run_id=rf_version.run_id, artifact_path="tfidf_vectorizer.pkl"
    )
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "'text' field must not be empty"}), 400

    result = predict(text, vectorizer, rf_model)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
