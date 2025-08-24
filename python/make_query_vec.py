import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_FILE = Path("cpp/query.npy")

if len(sys.argv) < 2:
    print("Usage: python make_query_vec.py 'your query here'")
    sys.exit(1)

query_text = sys.argv[1]
model = SentenceTransformer(MODEL_NAME)
embedding = model.encode(query_text).astype("float32")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_FILE, embedding)
print(f"Query vector saved → {OUT_FILE}")
