import type { FrameData } from "@/types";

/**
 * Dummy frame data for overlay rendering demo.
 * Coordinates are in video-native pixel space (e.g., 1280x720).
 */
export const DUMMY_FRAME_DATA: FrameData = {
  frameNumber: 120,
  timestamp: 4.0,
  holds: [
    { id: "h1", x: 200, y: 150, width: 60, height: 50, confidence: 0.95 },
    { id: "h2", x: 400, y: 280, width: 55, height: 45, confidence: 0.88 },
    { id: "h3", x: 350, y: 100, width: 50, height: 40, confidence: 0.92 },
    { id: "h4", x: 550, y: 200, width: 65, height: 55, confidence: 0.78 },
    { id: "h5", x: 300, y: 400, width: 58, height: 48, confidence: 0.85 },
  ],
  pose: [
    { name: "nose", x: 380, y: 180, confidence: 0.96 },
    { name: "left_shoulder", x: 350, y: 230, confidence: 0.94 },
    { name: "right_shoulder", x: 410, y: 230, confidence: 0.93 },
    { name: "left_elbow", x: 310, y: 200, confidence: 0.91 },
    { name: "right_elbow", x: 450, y: 195, confidence: 0.89 },
    { name: "left_wrist", x: 280, y: 170, confidence: 0.87 },
    { name: "right_wrist", x: 480, y: 175, confidence: 0.86 },
    { name: "left_hip", x: 360, y: 350, confidence: 0.92 },
    { name: "right_hip", x: 400, y: 350, confidence: 0.91 },
    { name: "left_knee", x: 350, y: 430, confidence: 0.88 },
    { name: "right_knee", x: 410, y: 435, confidence: 0.87 },
    { name: "left_ankle", x: 340, y: 510, confidence: 0.85 },
    { name: "right_ankle", x: 420, y: 515, confidence: 0.84 },
  ],
  predictions: [
    { holdId: "h3", probability: 0.82 },
    { holdId: "h4", probability: 0.45 },
  ],
};
