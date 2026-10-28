import pandas as pd
from neo4j import GraphDatabase

# Dataset location
CSV_PATH = "data/processed_qa_data.csv"

# Neo4j connection details
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4j"

# Read the CSV file
df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully")
print("Total rows:", len(df))


def create_graph(tx, row_id, knowledge, question, answer, label):
    tx.run(
        """
        MERGE (k:Knowledge {knowledge_id: $row_id})
        SET k.text = $knowledge

        MERGE (q:Question {question_id: $row_id})
        SET q.text = $question

        MERGE (a:Answer {answer_id: $row_id})
        SET a.text = $answer,
            a.label = $label

        MERGE (k)-[:HAS_QUESTION]->(q)
        MERGE (q)-[:HAS_ANSWER]->(a)
        """,
        row_id=int(row_id),
        knowledge=knowledge,
        question=question,
        answer=answer,
        label=int(label)
    )


# Connect to Neo4j
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

try:
    driver.verify_connectivity()
    print("Connected to Neo4j successfully")

    # Use your database name
    with driver.session(database="llmverification") as session:
        for index, row in df.iterrows():
            session.execute_write(
                create_graph,
                index,
                str(row["knowledge"]),
                str(row["question"]),
                str(row["answer"]),
                int(row["label"])
            )

    print("Dataset imported into Neo4j successfully")

except Exception as error:
    print("Error:", error)

finally:
    driver.close()