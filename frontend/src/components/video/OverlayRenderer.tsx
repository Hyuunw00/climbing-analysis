"use client";

import { useEffect, useRef, useCallback } from "react";
import type { FrameData, Hold, PoseKeypoint, Prediction } from "@/types";

// Pose skeleton connections (COCO format pairs)
const SKELETON_CONNECTIONS: [string, string][] = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

interface OverlayRendererProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  frameData: FrameData | null;
  isPlaying: boolean;
}

export default function OverlayRenderer({
  canvasRef,
  videoRef,
  frameData,
  isPlaying,
}: OverlayRendererProps) {
  const animFrameRef = useRef<number>(0);

  const syncCanvasSize = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const { clientWidth, clientHeight } = video;
    if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
      canvas.width = clientWidth;
      canvas.height = clientHeight;
    }
  }, [canvasRef, videoRef]);

  const drawHolds = useCallback(
    (ctx: CanvasRenderingContext2D, holds: Hold[], scaleX: number, scaleY: number) => {
      ctx.strokeStyle = "#4444ff";
      ctx.lineWidth = 2;
      ctx.font = "12px monospace";
      ctx.fillStyle = "#4444ff";

      for (const hold of holds) {
        const x = hold.x * scaleX;
        const y = hold.y * scaleY;
        const w = hold.width * scaleX;
        const h = hold.height * scaleY;

        ctx.strokeRect(x, y, w, h);
        // Label with confidence
        const label = `hold ${(hold.confidence * 100).toFixed(0)}%`;
        ctx.fillText(label, x, y - 4);
      }
    },
    []
  );

  const drawPose = useCallback(
    (ctx: CanvasRenderingContext2D, pose: PoseKeypoint[], scaleX: number, scaleY: number) => {
      const visible = pose.filter((kp) => kp.confidence >= 0.3);
      if (visible.length === 0) return;

      // Draw person bounding box
      const xs = visible.map((kp) => kp.x * scaleX);
      const ys = visible.map((kp) => kp.y * scaleY);
      const pad = 16;
      const minX = Math.min(...xs) - pad;
      const minY = Math.min(...ys) - pad;
      const maxX = Math.max(...xs) + pad;
      const maxY = Math.max(...ys) + pad;

      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;
      ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

      // Label
      ctx.font = "bold 13px monospace";
      ctx.fillStyle = "#00ff88";
      ctx.fillText("CLIMBER", minX + 4, minY - 6);

      // Draw connections
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;

      for (const [nameA, nameB] of SKELETON_CONNECTIONS) {
        const a = pose.find((kp) => kp.name === nameA);
        const b = pose.find((kp) => kp.name === nameB);
        if (!a || !b || a.confidence < 0.3 || b.confidence < 0.3) continue;

        ctx.beginPath();
        ctx.moveTo(a.x * scaleX, a.y * scaleY);
        ctx.lineTo(b.x * scaleX, b.y * scaleY);
        ctx.stroke();
      }

      // Draw joints
      ctx.fillStyle = "#00ff88";
      for (const kp of pose) {
        if (kp.confidence < 0.3) continue;
        ctx.beginPath();
        ctx.arc(kp.x * scaleX, kp.y * scaleY, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    },
    []
  );

  const drawPredictions = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      predictions: Prediction[],
      holds: Hold[],
      pose: PoseKeypoint[] | null,
      scaleX: number,
      scaleY: number
    ) => {
      if (!pose || predictions.length === 0) return;

      // Find hand keypoints
      const leftWrist = pose.find((kp) => kp.name === "left_wrist");
      const rightWrist = pose.find((kp) => kp.name === "right_wrist");
      const handPoint = leftWrist && leftWrist.confidence > 0.3 ? leftWrist : rightWrist;
      if (!handPoint || handPoint.confidence < 0.3) return;

      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "#ffff00";
      ctx.lineWidth = 2;
      ctx.font = "bold 13px monospace";
      ctx.fillStyle = "#ffff00";

      for (const pred of predictions) {
        const targetHold = holds.find((h) => h.id === pred.holdId);
        if (!targetHold) continue;

        const startX = handPoint.x * scaleX;
        const startY = handPoint.y * scaleY;
        const endX = (targetHold.x + targetHold.width / 2) * scaleX;
        const endY = (targetHold.y + targetHold.height / 2) * scaleY;

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Probability label
        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;
        ctx.fillText(`${(pred.probability * 100).toFixed(0)}%`, midX + 4, midY - 4);
      }

      ctx.setLineDash([]);
    },
    []
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!frameData) return;

    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const cw = canvas.width;
    const ch = canvas.height;

    // Calculate actual video rendering area (handles letterboxing from object-fit: contain)
    const videoAspect = vw / vh;
    const canvasAspect = cw / ch;

    let renderW: number, renderH: number, offsetX: number, offsetY: number;
    if (canvasAspect > videoAspect) {
      // Pillarboxed (black bars on sides)
      renderH = ch;
      renderW = ch * videoAspect;
      offsetX = (cw - renderW) / 2;
      offsetY = 0;
    } else {
      // Letterboxed (black bars top/bottom)
      renderW = cw;
      renderH = cw / videoAspect;
      offsetX = 0;
      offsetY = (ch - renderH) / 2;
    }

    const scaleX = renderW;
    const scaleY = renderH;

    // Save context and translate to video rendering origin
    ctx.save();
    ctx.translate(offsetX, offsetY);

    drawHolds(ctx, frameData.holds, scaleX, scaleY);

    if (frameData.pose) {
      drawPose(ctx, frameData.pose, scaleX, scaleY);
    }

    drawPredictions(ctx, frameData.predictions, frameData.holds, frameData.pose, scaleX, scaleY);

    ctx.restore();
  }, [canvasRef, videoRef, frameData, syncCanvasSize, drawHolds, drawPose, drawPredictions]);

  useEffect(() => {
    const loop = () => {
      draw();
      animFrameRef.current = requestAnimationFrame(loop);
    };

    if (isPlaying || frameData) {
      animFrameRef.current = requestAnimationFrame(loop);
    }

    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [draw, isPlaying, frameData]);

  // Re-draw on resize
  useEffect(() => {
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  return null; // Renders to canvas via ref, no DOM output
}
