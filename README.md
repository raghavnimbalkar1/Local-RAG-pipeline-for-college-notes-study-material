# StudyBuddy - Local RAG Assistant
## A fully local Retrieval-Augmented Generation (RAG) system that allows you to query your personal documents using open-source LLMs.

## Features
Document Processing: Ingest PDFs/PPTs and chunk them for processing

**Local Embeddings**: Uses SentenceTransformers to create embeddings

**Vector Storage**: FAISS for efficient similarity search

**Local LLM**: Runs completely offline with Llama 8B via LM Studio

**Citation Support**: Shows exactly which parts of your notes were used for answers

## Tech Stack
**Frontend**: React.js

**Backend**: Python (FastAPI)

**Embeddings**: SentenceTransformers

**Vector Database**: FAISS with C++ wrapper

**LLM**: Llama 8B via LM Studio

## How It Works
Upload your study materials (PDFs/PPTs)

Documents are chunked and embedded using SentenceTransformers

Embeddings are stored in FAISS for fast retrieval

When you ask a question, relevant chunks are retrieved

Llama 8B model generates answers based on retrieved context

Response includes citations to original source material


## Setting Up
### 1. Clone/Download the Repository

### 2. Install Dependencies:
`$ npm install`

### 3. Start the Application
`$ npm run start`
`$ python3 app.py`

### Requires local LM Studio server running on port 8000.
