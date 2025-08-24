import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Paths
FAISS_FILE = Path("index/notes.faiss")
ID_MAP_FILE = Path("index/id_map.jsonl")

# Load index
index = faiss.read_index(str(FAISS_FILE))

# Load ID map
id_map = [json.loads(line) for line in ID_MAP_FILE.open("r", encoding="utf-8")]

# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Example query
query_text = input("Enter query: ")
query_emb = model.encode(query_text).astype("float32").reshape(1, -1)

# Search top-k
k = 5
D, I = index.search(query_emb, k)

print("\nTop results:")
for score, idx in zip(D[0], I[0]):
    item = id_map[idx]
    print(f"[{score:.3f}] {item['source']} (page {item['page']}): {item['snippet']}")
