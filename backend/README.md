# Climbing Analytics Backend

FastAPI backend for video analysis (hold detection, pose tracking, route prediction).

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest tests/
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| POST | /api/analyze | Upload video for analysis |
| GET | /api/analyze/{job_id} | Check job status |
| GET | /api/analyze/{job_id}/result | Get analysis result |
