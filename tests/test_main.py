from fastapi.testclient import TestClient
from app.main import app


def test_root_endpoint():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert "RAG-ML API is running" in body["message"]
