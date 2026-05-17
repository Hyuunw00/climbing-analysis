from app.services.video_processor import extract_metadata

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
MAX_DURATION_SECONDS = 120  # 2 minutes


def validate_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def validate_duration(file_path: str) -> bool:
    """Reject videos longer than 2 minutes."""
    metadata = extract_metadata(file_path)
    if metadata is None:
        return False
    return 0 < metadata.duration <= MAX_DURATION_SECONDS
