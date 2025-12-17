# RAG-ML: Retrieval-Augmented Generation Demo

**RAG-ML** is a lightweight, production-oriented demonstration of a **Retrieval-Augmented Generation (RAG)** system.

It showcases how to:
* Ingest unstructured text
* Convert text into vector embeddings
* Store and search embeddings efficiently
* Retrieve relevant context for a query
* Generate grounded answers using an LLM

The project emphasizes **clarity, correctness, and real-world architecture** rather than framework abstraction or experimentation. It is designed to be easy to read, reason about, test, and deploy.

**Primary goals:**
* Demonstrate a clean RAG pipeline end to end
* Serve as a reference implementation
* Provide a solid foundation for extension into larger systems

**Non-goals:**
* Not a framework
* Not optimized for massive scale
* Not intended as a turnkey SaaS product

## Architecture & Design

The system implements a classic Retrieval-Augmented Generation (RAG) pipeline with a clean separation of concerns:

**1. Ingestion**
* Raw text is chunked using configurable size and overlap.
* Chunks are embedded via an embedding model.
* Embeddings are stored in a FAISS vector index alongside document metadata.

**2. Retrieval**
* Incoming queries are embedded using the same model.
* Top-K semantically similar chunks are retrieved from FAISS.

**3. Generation**
* Retrieved chunks are assembled into a context prompt.
* An LLM generates a grounded response using this context.

The backend is exposed via a FastAPI REST API and serves a minimal static frontend for manual testing. State (vector index and metadata) is persisted locally for simplicity.

## Tech Stack

* **Backend**: Python 3.11, FastAPI
* **Vector** Search: FAISS
* **Embeddings**: Sentence-Transformers / OpenAI embeddings
* **LLM**: OpenAI API
* **Frontend**: Static HTML/CSS/JavaScript
* **Dependency Management**: Poetry
* **Testing**: Pytest
* **Containerization**: Docker
* **Deployment**: Fly.io

## Getting Started

**Prerequisites**
* Python 3.11+
* Poetry
* An OpenAI key

**Installation**
* `git clone git@github.com:meirgarfinkel/rag-ml.git`
* `cd rag-ml`
* `poetry install`
* Create `.env` file in root of project: `touch .env`
* Add `OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxx` to the `.env` file.
* `poetry run uvicorn app.main:app --reload`
* Access the frontend at http://localhost:8000/

## License

This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.
