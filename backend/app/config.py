from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    upload_dir: str = "./uploads"
    max_file_size: int = 104857600  # 100MB
    frame_rate: int = 2

    # Hold detection model
    hold_model_path: str = "models/yolov8n-freeclimbs-detect-2.pt"
    hold_confidence_threshold: float = 0.5

    # URL download
    download_timeout: int = 60  # seconds

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
