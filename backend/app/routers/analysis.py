import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.config import settings
from app.models.schemas import AnalysisResponse, AnalysisResult, FrameData
from app.utils.video import validate_extension, validate_duration
from app.services.video_processor import process_video

router = APIRouter(prefix="/analyze", tags=["analysis"])

# In-memory job store (replace with DB/Redis later)
_jobs: dict[str, dict] = {}


def _run_analysis(job_id: str, file_path: str) -> None:
    """Run video analysis in background."""
    _jobs[job_id]["status"] = "processing"
    result = process_video(file_path)

    if "error" in result:
        _jobs[job_id].update({"status": "failed", "progress": 0.0})
        return

    _jobs[job_id].update({
        "status": "completed",
        "progress": 100.0,
        "result": result,
    })


@router.post("", response_model=AnalysisResponse)
async def create_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> AnalysisResponse:
    """Upload a video file and start analysis."""
    if not file.filename or not validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: mp4, mov, webm",
        )

    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_file_size:
        raise HTTPException(status_code=400, detail="File too large. Max 100MB")

    # Save file
    os.makedirs(settings.upload_dir, exist_ok=True)
    job_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[-1].lower()
    file_path = os.path.join(settings.upload_dir, f"{job_id}.{ext}")

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    # Validate duration
    if not validate_duration(file_path):
        os.remove(file_path)
        raise HTTPException(
            status_code=400, detail="Video too long. Max 2 minutes"
        )

    # Start background analysis
    _jobs[job_id] = {"status": "pending", "progress": 0.0, "file_path": file_path}
    background_tasks.add_task(_run_analysis, job_id, file_path)

    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Analysis started",
    )


@router.get("/{job_id}", response_model=AnalysisResponse)
async def get_analysis_status(job_id: str) -> AnalysisResponse:
    """Get analysis job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return AnalysisResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message="",
    )


@router.get("/{job_id}/result", response_model=AnalysisResult)
async def get_analysis_result(job_id: str) -> AnalysisResult:
    """Get analysis result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not yet completed")

    result = job["result"]
    frames = [FrameData(**f) for f in result["frames"]]

    return AnalysisResult(
        job_id=job_id,
        total_frames=len(frames),
        fps=result["metadata"]["fps"],
        duration=result["metadata"]["duration"],
        frames=frames,
        overlay_video_url=None,
    )
