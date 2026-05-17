"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FrameData, Hold, PoseKeypoint, Prediction, ForceDistribution } from "@/types";
import OverlayRenderer from "./OverlayRenderer";

const WS_URL = "ws://localhost:8000/api/realtime";
const CAPTURE_FPS = 3;
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY_MS = 2000;
const JPEG_QUALITY = 0.7;

type CameraState = "initializing" | "active" | "error" | "stopped";
type WsState = "connecting" | "connected" | "disconnected" | "error";

// Convert snake_case keys to camelCase recursively
function snakeToCamel(obj: unknown): unknown {
  if (Array.isArray(obj)) return obj.map(snakeToCamel);
  if (obj !== null && typeof obj === "object") {
    return Object.entries(obj as Record<string, unknown>).reduce(
      (acc, [key, value]) => {
        const camelKey = key.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
        acc[camelKey] = snakeToCamel(value);
        return acc;
      },
      {} as Record<string, unknown>,
    );
  }
  return obj;
}

interface CameraCaptureProps {
  onFrameData: (frame: FrameData) => void;
  onStop: () => void;
}

export default function CameraCapture({ onFrameData, onStop }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>("initializing");
  const [wsState, setWsState] = useState<WsState>("disconnected");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [frameData, setFrameData] = useState<FrameData | null>(null);

  // Connect to WebSocket
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setWsState("connecting");
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setWsState("connected");
      reconnectCountRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        const data = snakeToCamel(raw) as FrameData;
        setFrameData(data);
        onFrameData(data);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setWsState("disconnected");
      // Auto-reconnect
      if (reconnectCountRef.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectCountRef.current += 1;
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWs();
        }, RECONNECT_DELAY_MS);
      } else {
        setWsState("error");
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [onFrameData]);

  // Capture frame and send via WebSocket
  const captureAndSend = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!video || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    // Lazily create offscreen canvas
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
    }
    const canvas = captureCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    // Strip the data URL prefix to get pure base64
    const base64 = dataUrl.split(",")[1];
    if (base64) {
      ws.send(base64);
    }
  }, []);

  // Start frame capture loop
  const startCapture = useCallback(() => {
    if (captureIntervalRef.current) return;
    captureIntervalRef.current = setInterval(captureAndSend, 1000 / CAPTURE_FPS);
  }, [captureAndSend]);

  // Stop frame capture loop
  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  }, []);

  // Initialize camera
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setCameraState("active");
      } catch (err) {
        if (cancelled) return;
        setCameraState("error");
        if (err instanceof DOMException && err.name === "NotAllowedError") {
          setErrorMsg("Camera permission denied. Please allow camera access and try again.");
        } else {
          setErrorMsg("Failed to access camera. Make sure your device has a camera.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Connect WS and start capture when camera is active
  useEffect(() => {
    if (cameraState !== "active") return;

    connectWs();
    startCapture();

    return () => {
      stopCapture();
    };
  }, [cameraState, connectWs, startCapture, stopCapture]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Stop capture
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect
        wsRef.current.close();
      }
      // Stop camera
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  // Error state
  if (cameraState === "error") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 rounded-lg border border-border bg-panel p-8 min-h-[250px] sm:min-h-[300px]">
        <svg className="w-12 h-12 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M12 18.75H4.5a2.25 2.25 0 01-2.25-2.25V7.5A2.25 2.25 0 014.5 5.25h7.5" />
          <line x1="2" y1="2" x2="22" y2="22" strokeWidth={1.5} strokeLinecap="round" />
        </svg>
        <p className="text-sm text-red-400 text-center max-w-xs">{errorMsg}</p>
        <button
          type="button"
          onClick={onStop}
          className="text-sm sm:text-xs font-mono px-4 sm:px-3 py-2 sm:py-1.5 rounded border border-border hover:bg-border transition-colors text-muted hover:text-foreground"
        >
          Back to Upload
        </button>
      </div>
    );
  }

  // Initializing state
  if (cameraState === "initializing") {
    return (
      <div className="flex flex-1 items-center justify-center rounded-lg border border-border bg-panel min-h-[250px] sm:min-h-[300px]">
        <div className="flex flex-col items-center gap-3">
          <span className="inline-block h-3 w-3 rounded-full bg-accent animate-pulse" />
          <span className="text-sm text-muted font-mono">Requesting camera access...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-3 min-h-[250px] sm:min-h-[300px] lg:min-h-0">
      {/* Camera view with overlay */}
      <div className="relative flex-1 rounded-lg overflow-hidden bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-contain"
        />
        {/* Canvas overlay for pose/hold rendering */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        <OverlayRenderer
          canvasRef={canvasRef}
          videoRef={videoRef}
          frameData={frameData}
          isPlaying={true}
        />

        {/* LIVE badge */}
        {/* LIVE badge - larger on mobile for visibility */}
        <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm rounded px-2.5 sm:px-2 py-1.5 sm:py-1">
          <span className="relative flex h-2.5 sm:h-2 w-2.5 sm:w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-75" />
            <span className="relative inline-flex h-2.5 sm:h-2 w-2.5 sm:w-2 rounded-full bg-red-500" />
          </span>
          <span className="text-sm sm:text-xs font-bold tracking-wider text-red-400">LIVE</span>
        </div>

        {/* WebSocket status */}
        {wsState !== "connected" && (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm rounded px-2 py-1">
            <span className={`inline-block h-2 w-2 rounded-full ${
              wsState === "connecting" ? "bg-yellow-400 animate-pulse" :
              wsState === "error" ? "bg-red-500" : "bg-gray-500"
            }`} />
            <span className="text-xs font-mono text-muted">
              {wsState === "connecting" ? "Connecting..." :
               wsState === "error" ? "Connection failed" : "Disconnected"}
            </span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between rounded-lg bg-panel border border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <span className={`inline-block h-2 w-2 rounded-full ${
            wsState === "connected" ? "bg-green-500" :
            wsState === "connecting" ? "bg-yellow-400" : "bg-red-500"
          }`} />
          <span className="text-xs font-mono text-muted">
            {wsState === "connected" ? "Analyzing" :
             wsState === "connecting" ? "Connecting" : "Offline"}
          </span>
        </div>
        <button
          type="button"
          onClick={onStop}
          className="text-sm sm:text-xs font-mono px-4 sm:px-2 py-2 sm:py-1 min-h-[44px] sm:min-h-0 rounded border border-border hover:bg-border transition-colors text-muted hover:text-foreground"
        >
          Stop Camera
        </button>
      </div>
    </div>
  );
}
