# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import uvicorn
import json
import numpy as np
import faiss
from pathlib import Path
import requests
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI app ---
app = FastAPI()

# --- Enable CORS for React frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
INDEX_DIR = "../index"  # path to your index folder
CHAT_URL = "http://localhost:1234/v1/chat/completions"
TOP_K = 5

# --- Load FAISS index & metadata ---
index = faiss.read_index(str(Path(INDEX_DIR) / "notes.faiss"))

id_map = []
with open(Path(INDEX_DIR) / "id_map.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        id_map.append(json.loads(line))

# --- Load local embedding model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # must match index

# --- Request model ---
class QueryRequest(BaseModel):
    question: str

# --- API endpoint ---
@app.post("/query")
def query_rag_api(req: QueryRequest) -> Dict:
    question = req.question

    # Step 1: Embed query locally
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")

    # Step 2: FAISS search
    D, I = index.search(q_vec, TOP_K)

    # Step 3: Gather retrieved chunks
    retrieved = []
    for idx in I[0]:
        meta = id_map[idx]
        retrieved.append(f"[{meta['source']}] {meta.get('text','')}")
    context = "\n".join(retrieved)

    # Step 4: Build LM Studio prompt
    prompt = f"""You are a helpful assistant. 
Use the following retrieved notes to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    # Step 5: Call LM Studio
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
    answer = data["choices"][0]["message"]["content"]

    # Step 6: Return answer as JSON
    return {"answer": answer}

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
