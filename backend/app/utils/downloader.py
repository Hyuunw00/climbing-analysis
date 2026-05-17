"""Download videos from URLs using yt-dlp."""

import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

# Max file size after download (100MB)
MAX_DOWNLOAD_SIZE = 104_857_600
URL_PATTERN = re.compile(r"^https?://")


class DownloadError(Exception):
    """Raised when video download fails."""


def validate_url(url: str) -> None:
    """Ensure the URL uses http or https."""
    if not URL_PATTERN.match(url):
        raise DownloadError("Invalid URL. Only http/https URLs are supported.")


def download_video(url: str, output_dir: str, timeout: int = 60) -> str:
    """
    Download a video from a URL using yt-dlp.

    Args:
        url: Video URL (YouTube, Instagram, etc.)
        output_dir: Directory to save the downloaded file
        timeout: Download timeout in seconds

    Returns:
        Absolute path to the downloaded video file

    Raises:
        DownloadError: If download fails or yt-dlp is not available
    """
    validate_url(url)

    try:
        import yt_dlp
    except ImportError:
        raise DownloadError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )

    os.makedirs(output_dir, exist_ok=True)
    job_id = str(uuid.uuid4())
    output_template = os.path.join(output_dir, f"{job_id}.%(ext)s")

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": timeout,
        # Limit duration to 2 minutes
        "match_filter": _duration_filter,
        # Limit file size
        "max_filesize": MAX_DOWNLOAD_SIZE,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise DownloadError("Could not extract video info from URL.")

            # Find the downloaded file
            filename = ydl.prepare_filename(info)
            # yt-dlp may change extension after merge
            base, _ = os.path.splitext(filename)
            for ext in [".mp4", ".webm", ".mkv", ".mov"]:
                candidate = base + ext
                if os.path.exists(candidate):
                    logger.info("Downloaded video: %s", candidate)
                    return candidate

            # Fallback: check if exact filename exists
            if os.path.exists(filename):
                return filename

            raise DownloadError("Download completed but file not found.")

    except DownloadError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Unsupported URL" in error_msg:
            raise DownloadError(
                "Unsupported URL. Try YouTube, Instagram, or other major platforms."
            )
        if "Video unavailable" in error_msg or "Private video" in error_msg:
            raise DownloadError("Video is unavailable or private.")
        logger.exception("Download failed for URL: %s", url)
        raise DownloadError(f"Download failed: {error_msg}")


def _duration_filter(info_dict: dict, *, incomplete: bool = False) -> str | None:
    """Reject videos longer than 2 minutes."""
    duration = info_dict.get("duration")
    if duration and duration > 120:
        return "Video is longer than 2 minutes."
    return None
