import type { AnalysisStatus, AnalysisResult, FrameData } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Error ────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

// ── Raw backend shapes (snake_case) ──────────────────────────────────

interface RawAnalysisResponse {
  job_id: string;
  status: string;
  progress: number;
  message: string;
}

interface RawFrameData {
  frame_number: number;
  timestamp: number;
  holds: Array<{
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  }>;
  pose: Array<{
    name: string;
    x: number;
    y: number;
    z?: number;
    visibility?: number;
    confidence: number;
  }>;
  predictions: Array<{
    hold_id: string;
    probability: number;
    hand: string;
  }>;
}

interface RawAnalysisResult {
  job_id: string;
  total_frames: number;
  fps: number;
  duration: number;
  frames: RawFrameData[];
  overlay_video_url: string | null;
}

// ── Mappers ──────────────────────────────────────────────────────────

function mapStatus(raw: RawAnalysisResponse): AnalysisStatus {
  return {
    jobId: raw.job_id,
    status: raw.status as AnalysisStatus["status"],
    progress: raw.progress,
    message: raw.message,
  };
}

function mapFrame(raw: RawFrameData): FrameData {
  return {
    frameNumber: raw.frame_number,
    timestamp: raw.timestamp,
    holds: raw.holds,
    pose: raw.pose.map((kp) => ({
      name: kp.name,
      x: kp.x,
      y: kp.y,
      confidence: kp.confidence,
    })),
    predictions: raw.predictions.map((p) => ({
      holdId: p.hold_id,
      probability: p.probability,
      hand: p.hand,
    })),
  };
}

function mapResult(raw: RawAnalysisResult): AnalysisResult {
  return {
    jobId: raw.job_id,
    totalFrames: raw.total_frames,
    fps: raw.fps,
    duration: raw.duration,
    frames: raw.frames.map(mapFrame),
    overlayVideoUrl: raw.overlay_video_url,
  };
}

// ── API calls ────────────────────────────────────────────────────────

/**
 * Upload a video file and start analysis.
 * Returns the initial job status (with job_id).
 */
export async function uploadVideo(file: File): Promise<AnalysisStatus> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  const raw = await handleResponse<RawAnalysisResponse>(res);
  return mapStatus(raw);
}

/**
 * Poll the status/progress of an analysis job.
 */
export async function getAnalysisStatus(jobId: string): Promise<AnalysisStatus> {
  const res = await fetch(`${API_BASE}/api/analyze/${jobId}`);
  const raw = await handleResponse<RawAnalysisResponse>(res);
  return mapStatus(raw);
}

/**
 * Fetch the full analysis result once status is "completed".
 */
export async function getAnalysisResult(jobId: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/api/analyze/${jobId}/result`);
  const raw = await handleResponse<RawAnalysisResult>(res);
  return mapResult(raw);
}

/**
 * Get video file URL for a given job.
 */
export function getVideoUrl(jobId: string): string {
  return `${API_BASE}/api/analyze/${jobId}/video`;
}
