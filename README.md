# Hate Speech Detection - MLOps Pipeline

Binary hate speech classification (SVM & Random Forest) with TF-IDF (unigram)
features and SMOTE class balancing.

## Project Structure

```
hate-speech-mlops/
├── data/
│   ├── raw/             # original labeled_data.csv
│   └── processed/       # cleaned/lemmatized data
├── src/
│   ├── config.py        # paths & hyperparameters
│   ├── preprocessing.py # cleaning, tokenizing, lemmatizing
│   ├── train.py         # train + evaluate SVM and RF
│   └── predict.py       # load models, run inference
├── models/               # saved .pkl artifacts
├── notebooks/            # original EDA/dissertation notebook
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Place `labeled_data.csv` in `data/raw/`.
2. Train models:
   ```bash
   python -m src.train
   ```
3. Run inference:
   ```bash
   python -m src.predict
   ```
