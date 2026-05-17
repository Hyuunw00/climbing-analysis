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
    hand: str  # "left" or "right"


class ForceDistribution(BaseModel):
    """Estimated force distribution across contact points."""

    left_hand: float = 0.0
    right_hand: float = 0.0
    left_foot: float = 0.0
    right_foot: float = 0.0
    center_of_mass: dict[str, float] = {}


class FrameMetrics(BaseModel):
    """Performance metrics for a single frame."""

    smoothness: float = 50.0
    technique: float = 50.0
    stable: float = 50.0
    adhesion: float = 50.0
    score: float = 50.0
    vertical_velocity: float = 0.0


class FrameData(BaseModel):
    """Analysis data for a single frame."""

    frame_number: int
    timestamp: float
    holds: list[Hold] = []
    pose: list[PoseKeypoint] = []
    predictions: list[Prediction] = []
    force_distribution: ForceDistribution | None = None
    metrics: FrameMetrics | None = None


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
