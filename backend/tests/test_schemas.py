"""Tests for app.models.schemas — Pydantic model serialization."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    Hold,
    PoseKeypoint,
    Prediction,
    FrameData,
    AnalysisResponse,
    AnalysisResult,
)


class TestHold:
    def test_valid_hold(self):
        h = Hold(id="hold_0_1", x=100.5, y=200.3, width=50.0, height=60.0, confidence=0.95)
        assert h.id == "hold_0_1"
        assert h.confidence == 0.95

    def test_serialization_roundtrip(self):
        data = {"id": "h1", "x": 10.0, "y": 20.0, "width": 5.0, "height": 5.0, "confidence": 0.8}
        h = Hold(**data)
        assert h.model_dump() == data

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            Hold(id="h1", x=1.0, y=2.0)  # missing width, height, confidence


class TestPoseKeypoint:
    def test_defaults(self):
        kp = PoseKeypoint(name="nose", x=0.5, y=0.5)
        assert kp.z == 0.0
        assert kp.visibility == 0.0
        assert kp.confidence == 0.0

    def test_full_keypoint(self):
        kp = PoseKeypoint(name="left_wrist", x=0.3, y=0.7, z=-0.1, visibility=0.9, confidence=0.9)
        assert kp.name == "left_wrist"
        assert kp.z == -0.1

    def test_serialization(self):
        kp = PoseKeypoint(name="nose", x=0.5, y=0.5)
        d = kp.model_dump()
        assert "name" in d and "x" in d and "y" in d


class TestPrediction:
    def test_valid_prediction(self):
        p = Prediction(hold_id="hold_1", probability=0.85, hand="left")
        assert p.hold_id == "hold_1"
        assert p.hand == "left"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            Prediction(hold_id="hold_1")  # missing probability


class TestFrameData:
    def test_empty_frame(self):
        fd = FrameData(frame_number=0, timestamp=0.0)
        assert fd.holds == []
        assert fd.pose == []
        assert fd.predictions == []

    def test_frame_with_data(self):
        fd = FrameData(
            frame_number=5,
            timestamp=0.5,
            holds=[Hold(id="h1", x=1, y=2, width=3, height=4, confidence=0.9)],
            pose=[PoseKeypoint(name="nose", x=0.1, y=0.2)],
            predictions=[Prediction(hold_id="h1", probability=0.7, hand="right")],
        )
        assert len(fd.holds) == 1
        assert len(fd.pose) == 1
        assert len(fd.predictions) == 1


class TestAnalysisResponse:
    def test_basic_response(self):
        r = AnalysisResponse(job_id="abc-123", status="pending")
        assert r.progress == 0.0
        assert r.message == ""

    def test_full_response(self):
        r = AnalysisResponse(job_id="x", status="completed", progress=100.0, message="done")
        assert r.progress == 100.0


class TestAnalysisResult:
    def test_empty_result(self):
        r = AnalysisResult(job_id="j1")
        assert r.total_frames == 0
        assert r.frames == []
        assert r.overlay_video_url is None

    def test_json_roundtrip(self):
        r = AnalysisResult(job_id="j1", total_frames=2, fps=30.0, duration=5.0)
        json_str = r.model_dump_json()
        restored = AnalysisResult.model_validate_json(json_str)
        assert restored.job_id == "j1"
        assert restored.fps == 30.0
