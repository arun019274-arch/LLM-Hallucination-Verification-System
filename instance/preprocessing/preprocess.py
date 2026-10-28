import json
import csv
import os

# Get project folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# File paths
INPUT_FILE = os.path.join(BASE_DIR, "data", "qa_data.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed_qa_data.csv")


def preprocess_data():

    data = []

    # Read JSONL file line by line
    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                print("Skipping invalid line")

    processed_data = []

    for item in data:

        knowledge = str(item.get("knowledge", "")).strip()
        question = str(item.get("question", "")).strip()
        right_answer = str(item.get("right_answer", "")).strip()
        hallucinated_answer = str(
            item.get("hallucinated_answer", "")
        ).strip()

        # Skip incomplete records
        if not knowledge or not question:
            continue

        # Correct answer
        if right_answer:
            processed_data.append({
                "knowledge": knowledge,
                "question": question,
                "answer": right_answer,
                "label": 0
            })

        # Hallucinated answer
        if hallucinated_answer:
            processed_data.append({
                "knowledge": knowledge,
                "question": question,
                "answer": hallucinated_answer,
                "label": 1
            })

    # Save processed dataset
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "knowledge",
                "question",
                "answer",
                "label"
            ]
        )

        writer.writeheader()
        writer.writerows(processed_data)

    print("\nPreprocessing completed!")
    print("Original records:", len(data))
    print("Processed records:", len(processed_data))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    if __name__ == "__main__":preprocess_data()
