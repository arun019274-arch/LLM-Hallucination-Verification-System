
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_qa_data.csv"


def main():
    print("=" * 55)
    print("HALLUCINATION DETECTION MODEL EVALUATION")
    print("=" * 55)

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    required = {"answer", "label"}
    if not required.issubset(df.columns):
        print("Dataset must contain 'answer' and 'label' columns.")
        return

    df = df.dropna(subset=["answer", "label"]).copy()
    df["answer"] = df["answer"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df[df["label"].isin([0, 1])]

    if len(df) < 10 or df["label"].nunique() != 2:
        print("Not enough valid data or both classes are not present.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        df["answer"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=100000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions) * 100

    print(f"\nDataset records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"\nMeasured test accuracy: {accuracy:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Correct Answer", "Hallucinated Answer"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nEvaluation complete.")
    

if __name__ == "__main__":
    main()
