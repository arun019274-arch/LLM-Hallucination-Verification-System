import pandas as pd
import networkx as nx
from pyvis.network import Network
import re
import pickle


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "data/processed_qa_data.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Total rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 2. CREATE KNOWLEDGE GRAPH
# ============================================================

G = nx.DiGraph()


# ============================================================
# 3. CLEAN TEXT
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    return str(text).strip()


def normalize_entity(text):

    text = clean_text(text)

    text = re.sub(r"[?.!,;:]+$", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. EXTRACT SUBJECT / RELATION / OBJECT
# ============================================================

def extract_fact(question, answer):

    question = clean_text(question)
    answer = normalize_entity(answer)

    subject = None
    relation = None
    object_value = None

    # --------------------------------------------------------
    # Pattern 1:
    # Who is the president of India?
    # Who is the prime minister of India?
    # --------------------------------------------------------

    match = re.search(
        r"who is the (?:president|prime minister|pm) of (.+?)\??$",
        question,
        re.IGNORECASE
    )

    if match:

        object_value = normalize_entity(match.group(1))
        subject = answer

        if "president" in question.lower():
            relation = "PRESIDENT_OF"
        else:
            relation = "PRIME_MINISTER_OF"


    # --------------------------------------------------------
    # Pattern 2:
    # What is the capital of France?
    #
    # France -> CAPITAL_OF -> Paris
    # --------------------------------------------------------

    elif re.search(
        r"what is the capital of (.+?)\??$",
        question,
        re.IGNORECASE
    ):

        match = re.search(
            r"what is the capital of (.+?)\??$",
            question,
            re.IGNORECASE
        )

        subject = normalize_entity(match.group(1))
        relation = "CAPITAL_OF"
        object_value = answer


    # --------------------------------------------------------
    # Pattern 3:
    # Where is France located?
    #
    # France -> LOCATED_IN -> Europe
    # --------------------------------------------------------

    elif re.search(
        r"where is (.+?) located\??$",
        question,
        re.IGNORECASE
    ):

        match = re.search(
            r"where is (.+?) located\??$",
            question,
            re.IGNORECASE
        )

        subject = normalize_entity(match.group(1))
        relation = "LOCATED_IN"
        object_value = answer


    # --------------------------------------------------------
    # Pattern 4:
    # When was Albert Einstein born?
    # --------------------------------------------------------

    elif re.search(
        r"when was (.+?) born\??$",
        question,
        re.IGNORECASE
    ):

        match = re.search(
            r"when was (.+?) born\??$",
            question,
            re.IGNORECASE
        )

        subject = normalize_entity(match.group(1))
        relation = "BORN_IN"
        object_value = answer


    # --------------------------------------------------------
    # Pattern 5:
    # Who founded Microsoft?
    #
    # Bill Gates -> FOUNDED -> Microsoft
    # --------------------------------------------------------

    elif re.search(
        r"who founded (.+?)\??$",
        question,
        re.IGNORECASE
    ):

        match = re.search(
            r"who founded (.+?)\??$",
            question,
            re.IGNORECASE
        )

        subject = answer
        relation = "FOUNDED"
        object_value = normalize_entity(match.group(1))


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    else:

        subject = answer
        relation = "RELATED_TO"
        object_value = normalize_entity(question)


    return subject, relation, object_value


# ============================================================
# 5. BUILD KNOWLEDGE GRAPH
# ============================================================

for index, row in df.iterrows():

    knowledge = clean_text(row["knowledge"])
    question = clean_text(row["question"])
    answer = clean_text(row["answer"])
    label = row["label"]


    # --------------------------------------------------------
    # Extract fact
    # --------------------------------------------------------

    subject, relation, object_value = extract_fact(
        question,
        answer
    )

    subject = normalize_entity(subject)
    object_value = normalize_entity(object_value)


    # --------------------------------------------------------
    # Skip invalid records
    # --------------------------------------------------------

    if not subject or not object_value:
        continue


    # ========================================================
    # ENTITY IDS
    # ========================================================

    subject_id = f"ENTITY::{subject.lower()}"
    object_id = f"ENTITY::{object_value.lower()}"


    # ========================================================
    # SUBJECT NODE
    # ========================================================

    if not G.has_node(subject_id):

        G.add_node(
            subject_id,
            node_type="Entity",
            name=subject,
            title=subject
        )


    # ========================================================
    # OBJECT NODE
    # ========================================================

    if not G.has_node(object_id):

        G.add_node(
            object_id,
            node_type="Entity",
            name=object_value,
            title=object_value
        )


    # ========================================================
    # RELATIONSHIP
    # ========================================================

    G.add_edge(
        subject_id,
        object_id,
        relation=relation,
        title=relation
    )


    # ========================================================
    # CLAIM NODE
    # ========================================================

    claim_id = f"CLAIM::{index}"

    G.add_node(
        claim_id,
        node_type="Claim",
        name=f"Claim {index}",
        title=question,
        verification_label=label
    )


    # Claim -> Subject

    G.add_edge(
        claim_id,
        subject_id,
        relation="ABOUT",
        title="ABOUT"
    )


    # Claim -> Object

    G.add_edge(
        claim_id,
        object_id,
        relation="ABOUT",
        title="ABOUT"
    )


    # ========================================================
    # KNOWLEDGE NODE
    # ========================================================

    knowledge_id = f"KNOWLEDGE::{index}"

    G.add_node(
        knowledge_id,
        node_type="Knowledge",
        name=f"Knowledge {index}",

        # Store complete knowledge text
        title=knowledge[:500],
        knowledge_text=knowledge
    )


    # Claim -> Knowledge

    G.add_edge(
        claim_id,
        knowledge_id,
        relation="SUPPORTED_BY",
        title="SUPPORTED_BY"
    )


# ============================================================
# 6. GRAPH STATISTICS
# ============================================================

print()
print("Knowledge graph created successfully")

print("Total nodes:", G.number_of_nodes())
print("Total relationships:", G.number_of_edges())


# ============================================================
# 7. KNOWLEDGE TEXT TEST
# ============================================================

print()
print("Knowledge Text Test")
print("-------------------")

knowledge_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "Knowledge"
]

print("Knowledge nodes:", len(knowledge_nodes))

if knowledge_nodes:

    sample_node = knowledge_nodes[0]

    sample_text = G.nodes[sample_node].get(
        "knowledge_text",
        ""
    )

    print("Sample knowledge:")
    print(sample_text[:300])


# ============================================================
# 8. GRAPH FACT TEST
# ============================================================

print()
print("Graph Fact Test")
print("----------------")

test_facts = [
    ("ENTITY::france", "ENTITY::paris", "CAPITAL_OF"),
    ("ENTITY::france", "ENTITY::europe", "LOCATED_IN")
]


for subject_id, object_id, relation in test_facts:

    if G.has_edge(subject_id, object_id):

        edge_data = G.get_edge_data(
            subject_id,
            object_id
        )

        if edge_data.get("relation") == relation:

            print(
                subject_id,
                "->",
                relation,
                "->",
                object_id,
                "FOUND"
            )

        else:

            print(
                subject_id,
                "->",
                object_id,
                "WRONG RELATION"
            )

    else:

        print(
            subject_id,
            "->",
            relation,
            "->",
            object_id,
            "NOT FOUND"
        )


# ============================================================
# 9. CREATE SMALL VISUALIZATION
# ============================================================

print()
print("Creating interactive visualization...")


# Visualize first 100 claims

claim_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "Claim"
][:100]


visualization_nodes = set(claim_nodes)


# Include neighbors

for claim in claim_nodes:

    visualization_nodes.update(
        G.successors(claim)
    )


H = G.subgraph(
    visualization_nodes
).copy()


print(
    "Visualization nodes:",
    H.number_of_nodes()
)

print(
    "Visualization relationships:",
    H.number_of_edges()
)


# ============================================================
# 10. PYVIS
# ============================================================

net = Network(
    height="800px",
    width="100%",
    directed=True,
    bgcolor="white"
)


net.from_nx(H)


# ============================================================
# 11. IMPROVE NODE LABELS
# ============================================================

for node in net.nodes:

    node_id = node["id"]

    data = H.nodes[node_id]

    node_type = data.get(
        "node_type",
        "Entity"
    )

    name = data.get(
        "name",
        node_id
    )


    # Entity

    if node_type == "Entity":

        node["label"] = name
        node["title"] = name
        node["shape"] = "dot"
        node["size"] = 25


    # Claim

    elif node_type == "Claim":

        node["label"] = name

        node["title"] = data.get(
            "title",
            name
        )

        node["shape"] = "box"
        node["size"] = 20


    # Knowledge

    elif node_type == "Knowledge":

        node["label"] = name

        node["title"] = data.get(
            "title",
            name
        )

        node["shape"] = "database"
        node["size"] = 20


# ============================================================
# 12. SHOW RELATIONSHIP NAMES
# ============================================================

for edge in net.edges:

    source = edge["from"]
    target = edge["to"]

    relation = H.edges[
        source,
        target
    ].get(
        "relation",
        "RELATED_TO"
    )

    edge["label"] = relation
    edge["title"] = relation
    edge["arrows"] = "to"


# ============================================================
# 13. PHYSICS SETTINGS
# ============================================================

net.set_options("""
{
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -3000,
      "centralGravity": 0.3,
      "springLength": 150,
      "springConstant": 0.04,
      "damping": 0.09
    },
    "minVelocity": 0.75
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true
  }
}
""")


# ============================================================
# 14. SAVE INTERACTIVE GRAPH
# ============================================================

output_file = "knowledge_graph.html"

net.write_html(output_file)


# ============================================================
# 15. SAVE NETWORKX GRAPH
# ============================================================

graph_file = "knowledge_graph.pkl"

with open(graph_file, "wb") as f:

    pickle.dump(
        G,
        f
    )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print()
print("Interactive knowledge graph created successfully")
print("File:", output_file)

print()
print("NetworkX graph saved successfully")
print("Graph file:", graph_file)