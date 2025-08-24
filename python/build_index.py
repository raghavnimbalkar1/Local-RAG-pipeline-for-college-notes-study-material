import os
import json
import numpy as np
import faiss
from pathlib import Path

# --- Config ---
PROCESSED_CHUNKS = "../data/processed/chunks.jsonl"   # processed text chunks
EMBEDDINGS_FILE = "../index/embeddings.npy"           # precomputed embeddings
INDEX_DIR = "index"

# --- Main ---
def main():
    Path(INDEX_DIR).mkdir(exist_ok=True)

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    dim = embeddings.shape[1]

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "notes.faiss"))

    print(f"✅ FAISS index built with {embeddings.shape[0]} vectors, dimension {dim}")

    # Load chunks metadata
    with open(PROCESSED_CHUNKS, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    # Write ID map (one line per chunk, ties embedding index to text)
    with open(os.path.join(INDEX_DIR, "id_map.jsonl"), "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            entry = {"id": i, "text": chunk["text"], "source": chunk.get("source", "unknown")}
            f.write(json.dumps(entry) + "\n")

    print("✅ ID map saved.")

if __name__ == "__main__":
    main()
