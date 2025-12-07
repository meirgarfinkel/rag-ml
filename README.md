# RAG-ML: Retrieval-Augmented Generation Demo

## Overview
A minimal but functional Retrieval-Augmented Generation (RAG) pipeline backend built with FastAPI and Python.
Users can query a document corpus with semantic search + LLM-generated answers via a simple REST API.

# Features
- Python 3.10+ with FastAPI async REST API
- RAG query endpoint (`/v1/query`) integrating vector search + LLM calls
- Minimal static HTML frontend for test queries (served from backend)
- Dependency management via Poetry, ensuring reproducible installs
- Clear project structure for easy extension and customization

## Getting Started


### Prerequisites
- Python 3.10 or newer
- Poetry for dependency management
- Pop!_OS, Ubuntu or macOS environments supported

### Installation
- In your terminal run: `git clone git@github.com:meirgarfinkel/rag-ml.git && cd rag-ml && poetry install --no-root`

## Usage
- `poetry run uvicorn main:app --reload --port 8000`
- Open your browser at http://localhost:8000/docs to explore the automatic Swagger UI docs.
- Send POST requests to `/v1/query` with JSON payload `{"question": "YOUR QUERY HERE"}` to get RAG answers.
