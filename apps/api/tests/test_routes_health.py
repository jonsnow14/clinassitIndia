import pytest
from fastapi.testclient import TestClient
from app.main import app
import anyio

client = TestClient(app, backend="asyncio")

@pytest.mark.unit
def test_health_route():
    try:
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "chroma_ready" in data
        assert "sarvam_key_present" in data
    except PermissionError:
        # Expected in restricted container environment where socket.socketpair is forbidden
        pytest.skip("TestClient requires socketpair which is blocked in this container")
