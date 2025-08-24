import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Paths
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMB_FILE = Path("index/embeddings.npy")
ID_MAP_FILE = Path("index/id_map.jsonl")
FAISS_FILE = Path("index/notes.faiss")

# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

embeddings = []
id_map = []

print("Embedding chunks...")
with CHUNKS_FILE.open("r", encoding="utf-8") as fin, ID_MAP_FILE.open("w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        text = rec["text"]
        embedding = model.encode(text).astype("float32")
        embeddings.append(embedding)
        # Save minimal info for reference
        id_map.append({
            "id": rec["id"],
            "source": rec["source"],
            "page": rec["page"],
            "snippet": text[:150]  # first 150 chars for quick reference
        })
        fout.write(json.dumps(id_map[-1], ensure_ascii=False) + "\n")

embeddings = np.stack(embeddings)
EMB_FILE.parent.mkdir(parents=True, exist_ok=True)
np.save(EMB_FILE, embeddings)

print(f"Embeddings saved → {EMB_FILE}")
print(f"ID map saved → {ID_MAP_FILE}")

# Build FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

FAISS_FILE.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(FAISS_FILE))
print(f"FAISS index saved → {FAISS_FILE}")

print("Phase 3 fully done ✅")
