"""
Preprocessing pipeline for hate speech detection.

Covers:
- Anonymizing tweets (removing usernames)
- Dropping duplicates
- Re-mapping class labels to binary 'label'
- Cleaning text (lowercasing, removing punctuation, numbers, hashtags, etc.)
- Tokenizing, removing stopwords, lemmatizing
"""
import re
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from . import config

# Ensure required NLTK resources are available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def anonymize(text: str) -> str:
    """Remove usernames (e.g. @someone) from text."""
    return re.sub(r"(@[^\s]+)", "", text)


def clean_text(text: str) -> str:
    """
    Lowercase, remove brackets/parentheses, numbers, extra whitespace,
    newlines, quotes, &amp;, hashtags, 'rt', punctuation, and 'httptco'.
    """
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\"+", "", text)
    text = re.sub(r"(\&amp\;)", "", text)
    text = re.sub(r"(#[^\s]+)", "", text)
    text = re.sub(r"(rt)", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"(httptco)", "", text)
    return text


def tokenize_remove_stopwords_lemmatize(text: str) -> str:
    """Tokenize, remove stopwords, lemmatize, and rejoin into a string."""
    tokens = nltk.word_tokenize(text)
    filtered = [t for t in tokens if t.lower() not in STOP_WORDS]
    lemmed = " ".join([LEMMATIZER.lemmatize(w) for w in filtered])
    return lemmed


def preprocess_text(text: str) -> str:
    """
    Full preprocessing for a single piece of text (used at inference time).
    Combines clean_text + tokenize/stopword removal/lemmatization.
    """
    cleaned = clean_text(text)
    return tokenize_remove_stopwords_lemmatize(cleaned)


def load_and_prepare_data(raw_csv_path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw labeled_data.csv and return a DataFrame with columns:
    ['lemmatized_tweet', 'label']
    Steps: anonymize -> drop duplicates -> remap labels -> clean -> lemmatize
    """
    df = pd.read_csv(raw_csv_path, index_col=0)

    # Anonymize
    df["anon_tweets"] = df["tweet"].apply(anonymize)
    df = df.drop(columns=["tweet"])

    # Drop duplicates
    df = df.drop_duplicates("anon_tweets")

    # Remap class -> binary label (1 = Hate Speech, 0 = Other)
    df["label"] = df["class"].map(config.CLASS_TO_LABEL_MAP)

    # Clean text
    df["cleaned_tweets"] = df["anon_tweets"].apply(clean_text)

    # Tokenize, remove stopwords, lemmatize
    df["lemmatized_tweet"] = df["cleaned_tweets"].apply(tokenize_remove_stopwords_lemmatize)

    return df[["lemmatized_tweet", "label"]]


if __name__ == "__main__":
    data = load_and_prepare_data()
    data.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {config.PROCESSED_DATA_PATH}")
    print(data["label"].value_counts())
