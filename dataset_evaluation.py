
import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR, "data", "processed_qa_data.csv"
)

_df = None
_vectorizer = None
_tfidf_matrix = None


def load_dataset():
    """Load and index the CSV directly, without a knowledge graph."""
    global _df, _vectorizer, _tfidf_matrix

    if _df is not None:
        return _df, _vectorizer, _tfidf_matrix

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH).fillna("")

    required = {"knowledge", "question", "answer"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(sorted(missing))}"
        )

    # Each dataset row becomes one searchable document.
    df["search_text"] = (
        df["knowledge"].astype(str) + " "
        + df["question"].astype(str) + " "
        + df["answer"].astype(str)
    ).str.strip()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200000
    )

    matrix = vectorizer.fit_transform(df["search_text"])

    _df = df
    _vectorizer = vectorizer
    _tfidf_matrix = matrix

    print(f"Loaded {len(df)} dataset records.")
    return _df, _vectorizer, _tfidf_matrix


def verify_claim_dataset_only(claim, threshold=0.15):
    """
    Retrieve relevant records using TF-IDF.
    This does not use the knowledge graph.
    """
    df, vectorizer, matrix = load_dataset()

    claim = str(claim).strip()

    if not claim:
        return {
            "claim": claim,
            "status": "UNVERIFIED",
            "similarity": 0.0,
            "evidence": "",
            "reason": "No claim was provided."
        }

    query_vector = vectorizer.transform([claim])
    scores = cosine_similarity(query_vector, matrix).ravel()

    best_index = int(scores.argmax())
    best_score = float(scores[best_index])
    best_row = df.iloc[best_index]

    evidence = str(best_row["knowledge"]).strip()
    if not evidence:
        evidence = str(best_row["answer"]).strip()

    # This threshold is only a retrieval threshold, not an accuracy level.
    if best_score < threshold:
        return {
            "claim": claim,
            "status": "UNVERIFIED",
            "similarity": round(best_score * 100, 2),
            "evidence": evidence,
            "reason": (
                "No sufficiently similar record was retrieved. "
                "This does not prove the claim false."
            )
        }

    return {
        "claim": claim,
        "status": "RECORD_FOUND",
        "similarity": round(best_score * 100, 2),
        "evidence": evidence,
        "reason": (
            "A potentially relevant record was retrieved. "
            "Check whether it actually supports or contradicts the claim."
        )
    }


def verify_text_dataset_only(text):
    """Check each non-empty line as a separate claim."""
    claims = [
        line.strip()
        for line in str(text).splitlines()
        if line.strip()
    ]

    results = [
        verify_claim_dataset_only(claim)
        for claim in claims
    ]

    return {
        "results": results,
        "total": len(results),
        "record_found": sum(
            item["status"] == "RECORD_FOUND"
            for item in results
        ),
        "unverified": sum(
            item["status"] == "UNVERIFIED"
            for item in results
        )
    }


if __name__ == "__main__":
    print("Dataset-only verification test")
    print("No knowledge graph is used.")

    try:
        claim = input("Enter one claim: ")
        result = verify_claim_dataset_only(claim)

        print("\nClaim:", result["claim"])
        print("Status:", result["status"])
        print("Retrieval similarity:", str(result["similarity"]) + "%")
        print("Evidence:", result["evidence"])
        print("Reason:", result["reason"])

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print("Dataset error:", error)
