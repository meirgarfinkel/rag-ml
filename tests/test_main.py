from fastapi.testclient import TestClient
from app.main import app


def test_root_endpoint_serves_html():
    client = TestClient(app)

    response = client.get("")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>RAG-ML</title>" in response.text

def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
