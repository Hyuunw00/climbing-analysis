"""Tests for app.utils.video — file validation utilities."""

from unittest.mock import patch, MagicMock

from app.utils.video import validate_extension, validate_duration, ALLOWED_EXTENSIONS
from app.services.video_processor import VideoMetadata


class TestValidateExtension:
    """Tests for validate_extension."""

    def test_valid_mp4(self):
        assert validate_extension("video.mp4") is True

    def test_valid_mov(self):
        assert validate_extension("video.mov") is True

    def test_valid_webm(self):
        assert validate_extension("video.webm") is True

    def test_uppercase_extension(self):
        assert validate_extension("video.MP4") is True

    def test_mixed_case_extension(self):
        assert validate_extension("video.Mp4") is True

    def test_invalid_extension_txt(self):
        assert validate_extension("file.txt") is False

    def test_invalid_extension_avi(self):
        assert validate_extension("file.avi") is False

    def test_no_extension(self):
        assert validate_extension("noextension") is False

    def test_empty_string(self):
        assert validate_extension("") is False

    def test_multiple_dots(self):
        assert validate_extension("my.video.file.mp4") is True

    def test_dot_only(self):
        assert validate_extension(".mp4") is True


class TestValidateDuration:
    """Tests for validate_duration."""

    @patch("app.utils.video.extract_metadata")
    def test_valid_short_video(self, mock_extract):
        mock_extract.return_value = VideoMetadata(
            duration=30.0, width=1920, height=1080, fps=30.0, frame_count=900
        )
        assert validate_duration("/fake/path.mp4") is True

    @patch("app.utils.video.extract_metadata")
    def test_valid_at_boundary(self, mock_extract):
        mock_extract.return_value = VideoMetadata(
            duration=120.0, width=1920, height=1080, fps=30.0, frame_count=3600
        )
        assert validate_duration("/fake/path.mp4") is True

    @patch("app.utils.video.extract_metadata")
    def test_too_long(self, mock_extract):
        mock_extract.return_value = VideoMetadata(
            duration=121.0, width=1920, height=1080, fps=30.0, frame_count=3630
        )
        assert validate_duration("/fake/path.mp4") is False

    @patch("app.utils.video.extract_metadata")
    def test_zero_duration(self, mock_extract):
        mock_extract.return_value = VideoMetadata(
            duration=0.0, width=1920, height=1080, fps=30.0, frame_count=0
        )
        assert validate_duration("/fake/path.mp4") is False

    @patch("app.utils.video.extract_metadata")
    def test_metadata_extraction_fails(self, mock_extract):
        mock_extract.return_value = None
        assert validate_duration("/fake/path.mp4") is False
