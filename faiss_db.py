import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load HR policy data
file_path = "cleaned_hr_policy.jsonl"
documents = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        entry = json.loads(line)
        documents.append((entry["instruction"], entry["response"]))

# Load sentence transformer model (free alternative to OpenAI embeddings)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to vectors
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Generate embeddings
embeddings = np.array([get_embedding(doc[0]) for doc in documents])

# Store in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "hr_policy_faiss.index")

# Save text-to-index mapping
with open("hr_policy_mapping.json", "w", encoding="utf-8") as f:
    json.dump(documents, f)

print("✅ HR policy stored in FAISS vector database!")
