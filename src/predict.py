"""
Inference module: load vectorizer and predict using the registered RF model.
"""
import pickle

from . import config
from .preprocessing import preprocess_text

label_map = {0: "Not Hate Speech", 1: "Hate Speech"}


def predict(text: str, vectorizer=None, rf_model=None) -> dict:
    if vectorizer is None or rf_model is None:
        raise ValueError("vectorizer and rf_model must be provided")

    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])

    rf_pred = rf_model.predict(vectorized)[0]
    rf_conf = float(max(rf_model.predict_proba(vectorized)[0]))

    return {
        "input": text,
        "cleaned": cleaned,
        "label": label_map.get(rf_pred, "Not Hate Speech"),
        "confidence": rf_conf,
        "model": "rf_model (MLflow Registry)",
    }


if __name__ == "__main__":
    import pickle
    import mlflow.sklearn
    import mlflow

    client = mlflow.MlflowClient()
    rf_version = client.get_registered_model("rf_model").latest_versions[0]
    rf_model = mlflow.sklearn.load_model("models:/rf_model/latest")

    import mlflow.artifacts
    vectorizer_path = mlflow.artifacts.download_artifacts(
        run_id=rf_version.run_id, artifact_path="tfidf_vectorizer.pkl"
    )
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    test_tweets = ["I Hate You", "I love you", "You are disgusting"]
    for tweet in test_tweets:
        result = predict(tweet, vectorizer, rf_model)
        print(f"{tweet} -> {result['label']} ({result['confidence']:.2%})")
