
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# trainmodel.py is inside:
# LLM-Verification-System/instance/training/

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed_qa_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["answer", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# Remove empty values
df = df.dropna(subset=["answer", "label"])


# Convert answer to string
df["answer"] = df["answer"].astype(str)

# Convert labels to integer
df["label"] = df["label"].astype(int)


# ============================================================
# 4. INPUT AND OUTPUT
# ============================================================

# We use the ANSWER text for the baseline classifier.
#
# label = 0 -> Correct answer
# label = 1 -> Hallucinated answer

X = df["answer"]
y = df["label"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. TF-IDF VECTORIZATION
# ============================================================

print("\nConverting text into TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print("\nTraining model...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Training completed!")


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Correct Answer",
            "Hallucinated Answer"
        ]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 10. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "hallucination_model.pkl"
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

with open(vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)


# ============================================================
# 11. CONFIRM FILES
# ============================================================

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print("Model:")
print(model_path)

print("\nTF-IDF Vectorizer:")
print(vectorizer_path)


# ============================================================
# 12. TEST WITH SAMPLE ANSWERS
# ============================================================

print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

sample_answers = [
    "Arthur's Magazine",
    "First for Women was started first.",
    "The head office of the Oberoi Group is located in Delhi.",
    "The head office of the Oberoi Group is located in Mumbai."
]

sample_tfidf = vectorizer.transform(sample_answers)

sample_predictions = model.predict(sample_tfidf)
sample_probabilities = model.predict_proba(sample_tfidf)

for text, prediction, probability in zip(
    sample_answers,
    sample_predictions,
    sample_probabilities
):

    if prediction == 0:
        result = "CORRECT"
    else:
        result = "HALLUCINATED"

    confidence = max(probability) * 100

    print("\nText:", text)
    print("Prediction:", result)
    print(f"Confidence: {confidence:.2f}%")

