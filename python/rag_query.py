import json
import numpy as np
import faiss
from pathlib import Path
import requests
from sentence_transformers import SentenceTransformer

# --- Config ---
INDEX_DIR = "index"
CHAT_URL = "http://localhost:1234/v1/chat/completions"
TOP_K = 5

# --- Load embedding model locally ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # same model used for indexing

# --- Helpers ---
def embed_texts(texts):
    """Embed texts locally with SentenceTransformer."""
    return embedder.encode(texts, convert_to_numpy=True)

def load_id_map():
    """Load metadata for each FAISS vector."""
    id_map = []
    with open(Path(INDEX_DIR) / "id_map.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            id_map.append(json.loads(line))
    return id_map

def query_rag(question, top_k=TOP_K):
    # --- Load FAISS index + metadata ---
    index = faiss.read_index(str(Path(INDEX_DIR) / "notes.faiss"))
    id_map = load_id_map()

    # --- Embed query ---
    q_vec = embed_texts([question]).astype("float32")

    # --- Search FAISS ---
    D, I = index.search(q_vec, top_k)

    # --- Gather retrieved context ---
    retrieved = []
    for idx in I[0]:
        meta = id_map[idx]
        retrieved.append(f"[{meta['source']}] {meta.get('text','')}")

    context = "\n".join(retrieved)

    # --- Build prompt for LM Studio ---
    prompt = f"""You are a helpful assistant. 
Use the following retrieved notes to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    # --- Call LM Studio ---
    response = requests.post(
        CHAT_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": "local-llama",  
            "messages": [
                {"role": "system", "content": "You are a knowledgeable tutor."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
    )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

# --- Run example ---
if __name__ == "__main__":
    question = "What is backpropagation?"
    answer = query_rag(question)
    print("Answer:", answer)
