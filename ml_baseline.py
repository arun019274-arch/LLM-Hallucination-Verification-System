
from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATASET = Path(__file__).parent / "data" / "processed_qa_data.csv"


def split_into_claims(text):
    """Split a paragraph into individual sentences."""
    claims = re.split(r'(?<=[.!?])\s+', text.strip())
    return [claim.strip() for claim in claims if claim.strip()]


def main():
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        return

    df = pd.read_csv(DATASET)

    if not {"answer", "label"}.issubset(df.columns):
        print("Dataset must contain answer and label columns.")
        return

    # Clean the dataset
    df = df.dropna(subset=["answer", "label"]).copy()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["answer"] != ""]
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    if df["label"].nunique() != 2:
        print("Expected both labels: 0 and 1.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        df["answer"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    # ML baseline model: no knowledge graph is used
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            max_features=50,
            min_df=5,
            max_df=0.95,
        )),
        ("classifier", LogisticRegression(
            C=0.001,
            max_iter=1000,
            random_state=42,
        )),
    ])

    print(f"Dataset records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print("\nTraining simplified ML baseline...")

    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    print("\n===== ML BASELINE EVALUATION =====")
    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Label 0", "Label 1"],
        zero_division=0,
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Predict each sentence separately
    print("\n===== PREDICT NEW TEXT CLAIM BY CLAIM =====")
    text = input("Enter a sentence or paragraph: ").strip()

    if not text:
        print("No text entered.")
        return

    claims = split_into_claims(text)

    if not claims:
        print("No claims detected.")
        return

    print(f"\nTotal claims detected: {len(claims)}")

    for index, claim in enumerate(claims, start=1):
        # Predict one claim
        label = int(model.predict([claim])[0])

        # Get the predicted label's probability
        probabilities = model.predict_proba([claim])[0]
        classes = list(model.named_steps["classifier"].classes_)
        confidence = probabilities[classes.index(label)] * 100

        if label == 1:
            status = "HALLUCINATION PREDICTED"
        else:
            status = "NOT HALLUCINATED"

        print(f"\nClaim {index}: {claim}")
        print(f"Prediction: {status}")
        print(f"Model confidence: {confidence:.2f}%")

    print("\nNote: These are ML predictions, not factual verification.")


if __name__ == "__main__":
    main()
