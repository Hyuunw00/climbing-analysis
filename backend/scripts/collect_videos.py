"""Download bouldering completion videos for ML training data.

Usage:
    python -m scripts.collect_videos --url-file data/video_urls.txt --output data/raw/videos/
    python -m scripts.collect_videos --url "https://youtube.com/..." --output data/raw/videos/
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: str) -> str | None:
    """Download a single video using yt-dlp.

    Returns the output file path, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", os.path.join(output_dir, "%(id)s.%(ext)s"),
        "--no-playlist",
        "--no-overwrites",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info("Downloaded: %s", url)
            return url
        else:
            logger.warning("Failed to download %s: %s", url, result.stderr[:200])
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Timeout downloading: %s", url)
        return None
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download bouldering videos for training")
    parser.add_argument("--url", type=str, help="Single video URL to download")
    parser.add_argument("--url-file", type=str, help="File with one URL per line")
    parser.add_argument("--output", type=str, default="data/raw/videos/", help="Output directory")
    args = parser.parse_args()

    urls: list[str] = []

    if args.url:
        urls.append(args.url)

    if args.url_file:
        if not os.path.exists(args.url_file):
            logger.error("URL file not found: %s", args.url_file)
            sys.exit(1)
        with open(args.url_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    if not urls:
        logger.error("No URLs provided. Use --url or --url-file")
        sys.exit(1)

    logger.info("Downloading %d videos to %s", len(urls), args.output)

    success = 0
    for i, url in enumerate(urls, 1):
        logger.info("[%d/%d] %s", i, len(urls), url)
        if download_video(url, args.output):
            success += 1

    logger.info("Done: %d/%d downloaded successfully", success, len(urls))


if __name__ == "__main__":
    main()
