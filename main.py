from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/v1/query")
async def rag_query(request: QueryRequest):
    return {"answer": f"RAG response to: {request.question}", "sources": []}

@app.get("/")
async def root():
    return {"message": "RAG-ML API ready - POST to /v1/query"}
