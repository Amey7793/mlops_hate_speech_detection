"""
Central configuration for paths and hyperparameters.
"""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "labeled_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "tweet_df.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")

# ---- Data split ----
TEST_SIZE = 0.3
CV_SIZE = 0.3
RANDOM_STATE = 35

# ---- Model hyperparameters ----
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "linear",
    "probability": True,
    "degree": 3,
    "gamma": "auto",
    "random_state": 22,
}

RF_PARAMS = {
    "n_estimators": 99,
    "random_state": 42,
}

# ---- MLflow ----
MLFLOW_EXPERIMENT_NAME = "hate-speech-detection"

# ---- Class label mapping ----
# Original 'class': 0 - Hate Speech, 1 - Offensive, 2 - Neither
# New binary 'label': 1 - Hate Speech, 0 - Not Hate Speech
CLASS_TO_LABEL_MAP = {0: 1, 1: 0, 2: 0}
