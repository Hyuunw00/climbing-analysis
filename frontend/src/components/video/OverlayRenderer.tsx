"use client";

import { useEffect, useRef, useCallback } from "react";
import type { FrameData, Hold, PoseKeypoint, Prediction, ForceDistribution, Dab } from "@/types";

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
        const label = `${(hold.confidence * 100).toFixed(0)}%`;
        ctx.fillText(label, x, y - 4);
      }
    },
    []
  );

  const drawDabs = useCallback(
    (ctx: CanvasRenderingContext2D, dabs: Dab[], scaleX: number, scaleY: number) => {
      if (!dabs || dabs.length === 0) return;

      ctx.strokeStyle = "#ff0000";
      ctx.lineWidth = 3;
      ctx.font = "bold 14px monospace";
      ctx.fillStyle = "#ff0000";

      for (const dab of dabs) {
        const x = (dab.x - dab.width / 2) * scaleX;
        const y = (dab.y - dab.height / 2) * scaleY;
        const w = dab.width * scaleX;
        const h = dab.height * scaleY;

        ctx.strokeRect(x, y, w, h);
        ctx.fillText("DAB", x, y - 4);
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

  // Map limb names to pose keypoint names
  const LIMB_KEYPOINTS: Record<string, string> = {
    leftHand: "left_wrist",
    rightHand: "right_wrist",
    leftFoot: "left_ankle",
    rightFoot: "right_ankle",
  };

  /**
   * Interpolate green (#00ff88) → yellow (#ffff00) → red (#ff4444) based on value 0..1
   */
  const forceColor = useCallback((ratio: number): string => {
    const t = Math.max(0, Math.min(1, ratio));
    if (t < 0.5) {
      // green → yellow
      const s = t / 0.5;
      const r = Math.round(0x00 + (0xff - 0x00) * s);
      const g = Math.round(0xff);
      const b = Math.round(0x88 * (1 - s));
      return `rgb(${r},${g},${b})`;
    }
    // yellow → red
    const s = (t - 0.5) / 0.5;
    const r = 0xff;
    const g = Math.round(0xff * (1 - s) + 0x44 * s);
    const b = Math.round(0x44 * s);
    return `rgb(${r},${g},${b})`;
  }, []);

  const drawForceDistribution = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      fd: ForceDistribution,
      pose: PoseKeypoint[],
      scaleX: number,
      scaleY: number,
    ) => {
      const visible = pose.filter((kp) => kp.confidence >= 0.3);
      if (visible.length === 0) return;

      const limbs: { key: keyof typeof LIMB_KEYPOINTS; value: number }[] = [
        { key: "leftHand", value: fd.leftHand },
        { key: "rightHand", value: fd.rightHand },
        { key: "leftFoot", value: fd.leftFoot },
        { key: "rightFoot", value: fd.rightFoot },
      ];

      const maxRadius = 28;
      const minRadius = 8;

      for (const { key, value } of limbs) {
        const kpName = LIMB_KEYPOINTS[key];
        const kp = pose.find((p) => p.name === kpName);
        if (!kp || kp.confidence < 0.3) continue;

        const cx = kp.x * scaleX;
        const cy = kp.y * scaleY;
        const ratio = Math.max(0, Math.min(1, value)); // value is already 0-1
        const radius = minRadius + (maxRadius - minRadius) * ratio;
        const color = forceColor(ratio);

        // Semi-transparent filled circle
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = color.replace("rgb", "rgba").replace(")", ",0.35)");
        ctx.fill();

        // Border ring
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Percentage label
        const label = `${Math.round(value * 100)}%`;
        ctx.font = "bold 11px monospace";
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, cx, cy);
      }

      // Draw center of mass crosshair
      if (fd.centerOfMass) {
        const comX = fd.centerOfMass.x * scaleX;
        const comY = fd.centerOfMass.y * scaleY;
        const size = 8;

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;

        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(comX - size, comY);
        ctx.lineTo(comX + size, comY);
        ctx.stroke();

        // Vertical line
        ctx.beginPath();
        ctx.moveTo(comX, comY - size);
        ctx.lineTo(comX, comY + size);
        ctx.stroke();

        // Small circle at center
        ctx.beginPath();
        ctx.arc(comX, comY, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
      }

      // Reset text alignment
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    },
    [forceColor],
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
        ctx.fillText(`${pred.probability.toFixed(0)}%`, midX + 4, midY - 4);
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

    if (frameData.dabs) {
      drawDabs(ctx, frameData.dabs, scaleX, scaleY);
    }

    if (frameData.pose) {
      drawPose(ctx, frameData.pose, scaleX, scaleY);
    }

    drawPredictions(ctx, frameData.predictions, frameData.holds, frameData.pose, scaleX, scaleY);

    // Force distribution overlay
    if (frameData.forceDistribution && frameData.pose) {
      drawForceDistribution(ctx, frameData.forceDistribution, frameData.pose, scaleX, scaleY);
    }

    ctx.restore();
  }, [canvasRef, videoRef, frameData, syncCanvasSize, drawHolds, drawDabs, drawPose, drawPredictions, drawForceDistribution]);

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
