"""Tests for app.services.video_processor — metadata and frame extraction."""

from unittest.mock import patch, MagicMock

import cv2 as cv2_real

from app.services.video_processor import extract_metadata, extract_frames, VideoMetadata


class TestExtractMetadata:
    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_valid_video(self, mock_cap_cls, mock_cv2_video):
        mock_cap_cls.return_value = mock_cv2_video

        result = extract_metadata("/fake/video.mp4")

        assert result is not None
        assert result.fps == 30.0
        assert result.frame_count == 150
        assert result.width == 1920
        assert result.height == 1080
        assert result.duration == 5.0  # 150 frames / 30 fps
        mock_cv2_video.release.assert_called_once()

    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_file_not_found(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        result = extract_metadata("/nonexistent/video.mp4")
        assert result is None

    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_zero_fps_returns_zero_duration(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2_real.CAP_PROP_FPS: 0.0,
            cv2_real.CAP_PROP_FRAME_COUNT: 100,
            cv2_real.CAP_PROP_FRAME_WIDTH: 640,
            cv2_real.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_cap_cls.return_value = mock_cap

        result = extract_metadata("/fake/video.mp4")
        assert result is not None
        assert result.duration == 0.0


class TestExtractFrames:
    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_basic_extraction(self, mock_cap_cls, mock_cv2_video):
        mock_cap_cls.return_value = mock_cv2_video

        frames = extract_frames("/fake/video.mp4", target_fps=5)

        # 30fps video, sample at 5fps => interval = 6
        # 150 frames, step by 6 => 25 samples
        assert len(frames) == 25
        assert frames[0] == (0, 0.0)
        assert frames[1] == (6, 0.2)  # 6 / 30 = 0.2

    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_file_not_found(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        frames = extract_frames("/nonexistent/video.mp4")
        assert frames == []

    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_zero_fps_returns_empty(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2_real.CAP_PROP_FPS: 0.0,
            cv2_real.CAP_PROP_FRAME_COUNT: 100,
            cv2_real.CAP_PROP_FRAME_WIDTH: 640,
            cv2_real.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_cap_cls.return_value = mock_cap

        frames = extract_frames("/fake/video.mp4")
        assert frames == []

    @patch("app.services.video_processor.cv2.VideoCapture")
    def test_high_target_fps_equals_every_frame(self, mock_cap_cls):
        """If target_fps >= video_fps, interval = 1 so every frame is sampled."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2_real.CAP_PROP_FPS: 10.0,
            cv2_real.CAP_PROP_FRAME_COUNT: 30,
            cv2_real.CAP_PROP_FRAME_WIDTH: 640,
            cv2_real.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_cap_cls.return_value = mock_cap

        frames = extract_frames("/fake/video.mp4", target_fps=30)
        # interval = max(1, int(10/30)) = max(1, 0) = 1 => all 30 frames
        assert len(frames) == 30
