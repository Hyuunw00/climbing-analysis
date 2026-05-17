export interface Hold {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
  name: string;
}

export interface Prediction {
  holdId: string;
  probability: number;
  hand?: string; // "left" or "right"
}

export interface FrameData {
  frameNumber: number;
  timestamp: number;
  holds: Hold[];
  pose: PoseKeypoint[] | null;
  predictions: Prediction[];
}

export interface AnalysisResult {
  jobId: string;
  totalFrames: number;
  fps: number;
  duration: number;
  frames: FrameData[];
  overlayVideoUrl: string | null;
}

export interface AnalysisStatus {
  jobId: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  message?: string;
}

export type AppState = "idle" | "uploading" | "analyzing" | "ready" | "error";
