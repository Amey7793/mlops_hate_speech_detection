"""
Training pipeline for SVM and Random Forest hate speech classifiers.

Uses unigram TF-IDF (best performing setup per dissertation experiments),
SMOTE for class balancing, trains final models on (train+cv) data,
evaluates on held-out test set, and saves vectorizer + models to disk.
"""
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from . import config
from .preprocessing import load_and_prepare_data


def get_train_test_split(df: pd.DataFrame):
    """Split data into train+cv (X_temp) and test sets."""
    X = df["lemmatized_tweet"]
    y = df["label"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )
    return X_temp, X_test, y_temp, y_test


def vectorize_and_resample(X_temp, X_test, y_temp):
    """Fit TF-IDF on X_temp, transform X_test, apply SMOTE to balance training data."""
    tfidf_vectorizer = TfidfVectorizer()
    X_temp_vectorized = tfidf_vectorizer.fit_transform(X_temp)
    X_test_vectorized = tfidf_vectorizer.transform(X_test)

    smote = SMOTE(sampling_strategy="auto", random_state=config.RANDOM_STATE)
    X_temp_resampled, y_temp_resampled = smote.fit_resample(X_temp_vectorized, y_temp)

    return tfidf_vectorizer, X_temp_resampled, y_temp_resampled, X_test_vectorized


def evaluate_model(model, X_test_vectorized, y_test, model_name="Model"):
    """Compute and print evaluation metrics for a trained model."""
    preds = model.predict(X_test_vectorized)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted"),
        "recall": recall_score(y_test, preds, average="weighted"),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
    }

    print(f"\n{model_name} Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    cm = confusion_matrix(y_test, preds)
    print(f"{model_name} Confusion Matrix:\n{cm}")

    return metrics


def train_svm(X_train_resampled, y_train_resampled):
    model = SVC(**config.SVM_PARAMS)
    model.fit(X_train_resampled, y_train_resampled)
    return model


def train_rf(X_train_resampled, y_train_resampled):
    model = RandomForestClassifier(**config.RF_PARAMS)
    model.fit(X_train_resampled, y_train_resampled)
    return model


def save_artifacts(vectorizer, svm_model, rf_model):
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    with open(config.VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(config.SVM_MODEL_PATH, "wb") as f:
        pickle.dump(svm_model, f)

    with open(config.RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)

    print(f"\nSaved vectorizer to {config.VECTORIZER_PATH}")
    print(f"Saved SVM model to {config.SVM_MODEL_PATH}")
    print(f"Saved RF model to {config.RF_MODEL_PATH}")


def main():
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print("Loading and preprocessing data...")
    df = load_and_prepare_data()

    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = get_train_test_split(df)

    print("Vectorizing (TF-IDF unigram) and resampling (SMOTE)...")
    vectorizer, X_temp_resampled, y_temp_resampled, X_test_vectorized = \
        vectorize_and_resample(X_temp, X_test, y_temp)

    print("Training SVM...")
    with mlflow.start_run(run_name="SVM"):
        mlflow.log_params(config.SVM_PARAMS)
        mlflow.log_param("test_size", config.TEST_SIZE)

        svm_model = train_svm(X_temp_resampled, y_temp_resampled)
        svm_metrics = evaluate_model(svm_model, X_test_vectorized, y_test, "SVM")

        mlflow.log_metrics(svm_metrics)
        mlflow.sklearn.log_model(svm_model, "svm_model", skops_trusted_types=["scipy.sparse._csr.csr_matrix"])
        mlflow.log_artifact(config.VECTORIZER_PATH) if os.path.exists(config.VECTORIZER_PATH) else None

    print("Training Random Forest...")
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params(config.RF_PARAMS)
        mlflow.log_param("test_size", config.TEST_SIZE)

        rf_model = train_rf(X_temp_resampled, y_temp_resampled)
        rf_metrics = evaluate_model(rf_model, X_test_vectorized, y_test, "Random Forest")

        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(
            rf_model,
            "rf_model",
            skops_trusted_types=["scipy.sparse._csr.csr_matrix"],
            registered_model_name="rf_model",
        )
        mlflow.log_artifact(config.VECTORIZER_PATH)

    save_artifacts(vectorizer, svm_model, rf_model)


if __name__ == "__main__":
    main()
