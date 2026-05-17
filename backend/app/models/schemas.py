from pydantic import BaseModel


class Hold(BaseModel):
    """Detected climbing hold."""

    id: str
    x: float
    y: float
    width: float
    height: float
    confidence: float


class PoseKeypoint(BaseModel):
    """Single pose keypoint from MediaPipe Pose."""

    name: str
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    confidence: float = 0.0


class Prediction(BaseModel):
    """Next hold prediction."""

    hold_id: str
    probability: float


class FrameData(BaseModel):
    """Analysis data for a single frame."""

    frame_number: int
    timestamp: float
    holds: list[Hold] = []
    pose: list[PoseKeypoint] = []
    predictions: list[Prediction] = []


class AnalysisResponse(BaseModel):
    """Response for analysis job creation and status."""

    job_id: str
    status: str  # pending | processing | completed | failed
    progress: float = 0.0
    message: str = ""


class AnalysisResult(BaseModel):
    """Full analysis result."""

    job_id: str
    total_frames: int = 0
    fps: float = 0.0
    duration: float = 0.0
    frames: list[FrameData] = []
    overlay_video_url: str | None = None
