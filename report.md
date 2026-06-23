# Hate Speech Detection - MLOps Project Report

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for detecting hate speech in tweets. It covers data preprocessing, model training, experiment tracking, model registration, REST API development, UI development, and containerization using Docker.

---

## 2. Project Structure

```
hate-speech-mlops/
├── data/
│   └── raw/
│       └── labeled_data.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svm_model.pkl
│   └── rf_model.pkl
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── mlruns/
├── app.py
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── mlflow.db
```

---

## 3. Dataset

- **Source:** `labeled_data.csv`
- **Original Labels:**
  - 0 → Hate Speech
  - 1 → Offensive Language
  - 2 → Neither
- **Binary Mapping:**
  - 1 → Hate Speech
  - 0 → Not Hate Speech

---

## 4. Configuration (`src/config.py`)

Central configuration file for all paths and hyperparameters.

| Parameter | Value |
|-----------|-------|
| TEST_SIZE | 0.3 |
| CV_SIZE | 0.3 |
| RANDOM_STATE | 35 |
| SVM Kernel | linear |
| SVM C | 1.0 |
| RF n_estimators | 99 |
| RF random_state | 42 |
| MLflow Experiment | hate-speech-detection |

---

## 5. Preprocessing Pipeline (`src/preprocessing.py`)

Steps applied to raw tweet data:

1. **Anonymization** — Remove usernames (`@someone`)
2. **Deduplication** — Drop duplicate tweets
3. **Label Remapping** — Convert 3-class to binary
4. **Text Cleaning** — Lowercase, remove punctuation, numbers, hashtags, URLs
5. **Tokenization** — Split into tokens using NLTK
6. **Stopword Removal** — Remove common English stopwords
7. **Lemmatization** — Reduce words to base form using WordNetLemmatizer

---

## 6. Model Training (`src/train.py`)

### Feature Extraction
- **TF-IDF Vectorizer** (unigram) fitted on training data

### Class Balancing
- **SMOTE** (Synthetic Minority Oversampling Technique) applied to handle class imbalance

### Models Trained

#### Support Vector Machine (SVM)
- Kernel: Linear
- Probability: True
- C: 1.0

#### Random Forest (RF)
- n_estimators: 99
- random_state: 42

### Training Command
```powershell
python -m src.train
```

---

## 7. MLflow Experiment Tracking

### Setup
```powershell
mlflow ui
```
UI available at: `http://localhost:5000`

### Tracked per Run
- **Parameters:** Model hyperparameters, test size
- **Metrics:** Accuracy, Precision, Recall, F1 Score (weighted)
- **Artifacts:** Trained model, TF-IDF vectorizer

### Model Registration
- RF model registered in MLflow Model Registry as **`rf_model`**
- Auto-registered on every training run via `registered_model_name="rf_model"` in `log_model()`

---

## 8. Inference (`src/predict.py`)

- Accepts raw text input
- Applies full preprocessing pipeline
- Vectorizes using TF-IDF
- Returns prediction label and confidence score

### Output Format
```json
{
  "input": "original text",
  "cleaned": "preprocessed text",
  "label": "Hate Speech",
  "confidence": 0.87,
  "model": "rf_model (MLflow Registry)"
}
```

---

## 9. Flask REST API (`app.py`)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict hate speech |

### Request Format
```json
{
  "text": "your text here"
}
```

### Run Locally
```powershell
python app.py
```
API available at: `http://localhost:8000`

### Model Loading
- **Locally:** Loads RF model from MLflow registry + vectorizer from MLflow artifact store
- **In Docker:** Loads RF model and vectorizer from `.pkl` files (`USE_PKL=true`)

---

## 10. Streamlit UI (`streamlit_app.py`)

- User enters text in a text area
- Clicks **Predict** button
- Displays:
  - Label (Hate Speech / Not Hate Speech)
  - Confidence score
  - Model name
  - Cleaned text (expandable)

### Run Locally
```powershell
streamlit run streamlit_app.py
```
UI available at: `http://localhost:8501`

---

## 11. Docker Containerization

### Dockerfile
- Base image: `python:3.11-slim`
- Installs all dependencies from `requirements.txt`
- Downloads NLTK resources at build time
- Single image reused across all services

### docker-compose.yml

Three services running from one image (`hate-speech-mlops:latest`):

| Service | Port | Description |
|---------|------|-------------|
| mlflow | 5000 | MLflow tracking server |
| flask-api | 8000 | Flask REST API |
| streamlit | 8501 | Streamlit UI |

### Volumes
- `./mlruns` — MLflow run artifacts
- `./mlflow.db` — MLflow SQLite backend
- `./models` — Trained model pkl files

### Run Full Stack
```powershell
docker-compose up --build
```

---

## 12. Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| ML Library | scikit-learn |
| Oversampling | imbalanced-learn (SMOTE) |
| NLP | NLTK |
| Experiment Tracking | MLflow |
| API Framework | Flask |
| UI Framework | Streamlit |
| Containerization | Docker + Docker Compose |

---

## 13. How to Run

### Option 1: Locally (3 terminals)
```powershell
# Terminal 1 - MLflow
mlflow ui

# Terminal 2 - Flask API
python app.py

# Terminal 3 - Streamlit
streamlit run streamlit_app.py
```

### Option 2: Docker (1 command)
```powershell
docker-compose up --build
```

### Access Points
| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| Flask API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |

---

## 14. MLOps Pipeline Flow

```
Raw Data
   ↓
Preprocessing (clean, tokenize, lemmatize)
   ↓
Feature Extraction (TF-IDF)
   ↓
Class Balancing (SMOTE)
   ↓
Model Training (SVM + Random Forest)
   ↓
MLflow Tracking (params, metrics, artifacts)
   ↓
Model Registry (rf_model registered)
   ↓
Flask API (serve predictions)
   ↓
Streamlit UI (user interface)
   ↓
Docker (containerized deployment)
```
