import io
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_check(client: AsyncClient):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_upload_invalid_extension(client: AsyncClient):
    file = io.BytesIO(b"fake content")
    resp = await client.post(
        "/api/analyze",
        files={"file": ("test.txt", file, "text/plain")},
    )
    assert resp.status_code == 400
    assert "Invalid file type" in resp.json()["detail"]


@pytest.mark.anyio
async def test_upload_valid_file(client: AsyncClient):
    # Minimal valid mp4-like file (won't pass duration check without real video)
    file = io.BytesIO(b"\x00" * 1024)
    resp = await client.post(
        "/api/analyze",
        files={"file": ("test.mp4", file, "video/mp4")},
    )
    # Will fail duration validation since it's not a real video
    assert resp.status_code == 400
    assert "Video too long" in resp.json()["detail"] or "Max 2 minutes" in resp.json()["detail"]
