"""Tests for API endpoints using FastAPI TestClient."""

import io
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient

from app.routers.analysis import _jobs


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear the in-memory job store between tests."""
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.mark.anyio
class TestHealthCheck:
    async def test_returns_ok(self, client: AsyncClient):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
class TestCreateAnalysis:
    async def test_rejects_invalid_extension(self, client: AsyncClient):
        file = io.BytesIO(b"fake")
        resp = await client.post(
            "/api/analyze",
            files={"file": ("test.txt", file, "text/plain")},
        )
        assert resp.status_code == 400
        assert "Invalid file type" in resp.json()["detail"]

    async def test_rejects_empty_filename(self, client: AsyncClient):
        file = io.BytesIO(b"fake")
        resp = await client.post(
            "/api/analyze",
            files={"file": ("", file, "video/mp4")},
        )
        # FastAPI may return 422 (no filename) or 400 (invalid extension)
        assert resp.status_code in (400, 422)

    async def test_rejects_file_too_large(self, client: AsyncClient):
        # Create a file larger than max_file_size (100MB)
        # We mock settings to use a small limit for testing
        with patch("app.routers.analysis.settings") as mock_settings:
            mock_settings.max_file_size = 10  # 10 bytes
            mock_settings.upload_dir = "/tmp/test_uploads"
            file = io.BytesIO(b"x" * 50)
            resp = await client.post(
                "/api/analyze",
                files={"file": ("test.mp4", file, "video/mp4")},
            )
            assert resp.status_code == 400
            assert "File too large" in resp.json()["detail"]

    @patch("app.routers.analysis.validate_duration", return_value=True)
    @patch("app.routers.analysis.process_video")
    async def test_successful_upload(self, mock_process, mock_duration, client: AsyncClient):
        mock_process.return_value = {"metadata": {"fps": 30, "duration": 5.0}, "frames": []}

        file = io.BytesIO(b"\x00" * 100)
        resp = await client.post(
            "/api/analyze",
            files={"file": ("climb.mp4", file, "video/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "job_id" in data

    @patch("app.routers.analysis.validate_duration", return_value=False)
    async def test_rejects_video_too_long(self, mock_duration, client: AsyncClient):
        file = io.BytesIO(b"\x00" * 100)
        resp = await client.post(
            "/api/analyze",
            files={"file": ("climb.mp4", file, "video/mp4")},
        )
        assert resp.status_code == 400
        assert "Video too long" in resp.json()["detail"]


@pytest.mark.anyio
class TestGetAnalysisStatus:
    async def test_not_found(self, client: AsyncClient):
        resp = await client.get("/api/analyze/nonexistent-id")
        assert resp.status_code == 404

    async def test_returns_job_status(self, client: AsyncClient):
        _jobs["test-job"] = {"status": "processing", "progress": 50.0}
        resp = await client.get("/api/analyze/test-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50.0


@pytest.mark.anyio
class TestGetAnalysisResult:
    async def test_not_found(self, client: AsyncClient):
        resp = await client.get("/api/analyze/nonexistent-id/result")
        assert resp.status_code == 404

    async def test_not_completed(self, client: AsyncClient):
        _jobs["test-job"] = {"status": "processing", "progress": 50.0}
        resp = await client.get("/api/analyze/test-job/result")
        assert resp.status_code == 400
        assert "not yet completed" in resp.json()["detail"]

    async def test_returns_result(self, client: AsyncClient):
        _jobs["test-job"] = {
            "status": "completed",
            "progress": 100.0,
            "result": {
                "metadata": {"fps": 30.0, "duration": 5.0, "width": 1920, "height": 1080},
                "frames": [
                    {
                        "frame_number": 0,
                        "timestamp": 0.0,
                        "holds": [],
                        "pose": [],
                        "predictions": [],
                    }
                ],
            },
        }
        resp = await client.get("/api/analyze/test-job/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_frames"] == 1
        assert data["fps"] == 30.0
        assert data["duration"] == 5.0
