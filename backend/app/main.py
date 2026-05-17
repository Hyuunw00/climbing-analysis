from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analysis
from app.routers.realtime import router as realtime_router

app = FastAPI(title="Climbing Analytics API", version="0.1.0")

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(analysis.router, prefix="/api")
app.include_router(realtime_router, prefix="/api")


@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "ok"}
