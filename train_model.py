"""
train_model.py
Trains a Fake News Detector using TF-IDF + Logistic Regression.

Expects two CSV files in the data/ folder:
    data/Fake.csv
    data/True.csv

Both files must have at least a "title" and "text" column
(this matches the popular Kaggle "Fake and Real News Dataset" by Clement Bisaillon).

Run:
    python train_model.py
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/numbers/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"<.*?>", " ", text)                    # remove HTML tags
    text = re.sub(r"\d+", " ", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


def load_data():
    print("Loading data...")
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake["label"] = 0   # 0 = FAKE
    real["label"] = 1   # 1 = REAL

    df = pd.concat([fake, real], axis=0, ignore_index=True)

    # Combine title + text into one field (more signal for the model)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Shuffle rows so fake/real aren't in blocks
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    df = load_data()

    print("Cleaning text...")
    df["clean_content"] = df["content"].apply(clean_text)

    X = df["clean_content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,      # ignore words that appear in >70% of documents
        max_features=50000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nSaving model and vectorizer to model/ ...")
    joblib.dump(model, "model/fake_news_model.joblib")
    joblib.dump(vectorizer, "model/vectorizer.joblib")
    print("Done! Files saved: model/fake_news_model.joblib, model/vectorizer.joblib")


if __name__ == "__main__":
    main()
