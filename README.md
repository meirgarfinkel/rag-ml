# RAG-ML: Retrieval-Augmented Generation Demo

## Overview
A minimal but functional Retrieval-Augmented Generation (RAG) pipeline backend built with FastAPI and Python.
Users can query a document corpus with semantic search + LLM-generated answers via a simple REST API.

# Features
- Python 3.11+ with FastAPI async REST API
- RAG query endpoint (`/v1/query`) integrating vector search + LLM calls
- Minimal static HTML frontend for test queries (served from backend)
- Dependency management via Poetry, ensuring reproducible installs
- Clear project structure for easy extension and customization

## Getting Started


### Prerequisites
- Python 3.11 or newer
- Poetry for dependency management
- Pop!_OS, Ubuntu or macOS environments supported

### Installation
- In your terminal run: `git clone git@github.com:meirgarfinkel/rag-ml.git && cd rag-ml && poetry install --no-root`

## Usage
- `poetry run uvicorn main:app --reload --port 8000`
- Open your browser at http://localhost:8000/docs to explore the automatic Swagger UI docs.
- Send POST requests to `/v1/query` with JSON payload `{"question": "YOUR QUERY HERE"}` to get RAG answers.


1: Project Foundation and Dependency Management
2: Backend API Structure with FastAPI
3: Data Ingestion Pipeline
4: Vector Store Setup with FAISS
5: Embedding Model Integration
6: Retrieval Logic
7: Generation with OpenAI API
8: Frontend Interface
9: Containerization with Docker
10: Deployment on Fly.io


# Clone, install, configure, run (replace YOUR_API_KEY)
git clone git@github.com:meirgarfinkel/rag-ml.git && cd rag-ml
rm -f poetry.lock
pip install poetry
poetry install
echo "OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE" > .env
mkdir -p data
echo "Retrieval-Augmented Generation (RAG) combines retrieval of relevant documents with generative AI models.

RAG works by embedding documents into vectors using models like sentence-transformers/all-MiniLM-L6-v2. These vectors are stored in FAISS for fast similarity search.

Queries are embedded and matched against document vectors to retrieve relevant context. GPT-4o-mini then generates answers grounded in this context." > data/demo_dataset.txt && \
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# New terminal - load demo data
curl -X POST http://localhost:8000/api/v1/ingest/demo

# Ask a question
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3}'