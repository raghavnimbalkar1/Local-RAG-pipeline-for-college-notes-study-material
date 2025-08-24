import faiss
import torch
import langchain
import pypdf
import pptx
from sentence_transformers import SentenceTransformer

def safe_version(module, name):
    return getattr(module, "__version__", "unknown")

print("=== Environment Sanity Check ===")
print("FAISS version:", safe_version(faiss, "faiss"))
print("Torch version:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("LangChain version:", safe_version(langchain, "langchain"))
print("PyPDF version:", safe_version(pypdf, "pypdf"))
print("python-pptx version:", safe_version(pptx, "pptx"))

# Test loading embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully")
print("Embedding dimension:", model.get_sentence_embedding_dimension())
print("=================================")
