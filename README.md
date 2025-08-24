# Local RAG Chat

A **local retrieval-augmented generation (RAG) chatbot** for your personal notes, powered by **FAISS**, **SentenceTransformers**, and **LM Studio**. Ask questions in natural language, and the chatbot retrieves relevant notes and generates answers using your local LLaMA model.

---

## Features

- Local embeddings with **SentenceTransformer** (`all-MiniLM-L6-v2`).
- FAISS vector store for fast similarity search.
- Supports **RAG**: retrieves relevant notes and passes them as context to LLaMA.
- Frontend chat interface built with **React**, fully chat-like with bubbles.
- Works entirely offline once setup is complete (except downloading the model first).

---
