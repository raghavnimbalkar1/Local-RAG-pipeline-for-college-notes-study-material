# /python/make_query_vec.py
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Usage: python make_query_vec.py "your query here"
if len(sys.argv) < 2:
    print("Usage: python make_query_vec.py \"Your question here\"")
    sys.exit(1)

query_text = sys.argv[1]

# Load the same embedding model used in build_index.py
model = SentenceTransformer('all-MiniLM-L6-v2')  # replace with your model if different

# Compute embedding
embedding = model.encode([query_text], normalize_embeddings=True)  # shape (1, dim)

# Save to query.npy
np.save("query.npy", embedding)
print(f"Query vector saved to query.npy, dimension={embedding.shape[1]}")
