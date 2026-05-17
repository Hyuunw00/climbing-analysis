"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Header from "@/components/layout/Header";
import VideoUploader from "@/components/video/VideoUploader";
import VideoPlayer from "@/components/video/VideoPlayer";
import CameraCapture from "@/components/video/CameraCapture";
import DashboardPanel from "@/components/dashboard/DashboardPanel";
import { uploadVideo, uploadVideoUrl, getAnalysisStatus, getAnalysisResult, getVideoUrl, ApiError } from "@/lib/api";
import type { AnalysisResult, FrameData, AppState, AppMode } from "@/types";

const POLL_INTERVAL_MS = 1000;
const STORAGE_KEY = "climbing_analysis_job";

export default function Home() {
  const [appMode, setAppMode] = useState<AppMode>("upload");
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const [appState, setAppState] = useState<AppState>("idle");
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [currentFrame, setCurrentFrame] = useState<FrameData | null>(null);

  const jobIdRef = useRef<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Restore previous session on mount
  useEffect(() => {
    const savedJobId = localStorage.getItem(STORAGE_KEY);
    if (!savedJobId) return;

    (async () => {
      try {
        const status = await getAnalysisStatus(savedJobId);
        jobIdRef.current = savedJobId;

        if (status.status === "completed") {
          const result = await getAnalysisResult(savedJobId);
          setVideoSrc(getVideoUrl(savedJobId));
          setAnalysisResult(result);
          setAppState("ready");
          if (result.frames.length > 0) setCurrentFrame(result.frames[0]);
        } else if (status.status === "processing" || status.status === "pending") {
          setVideoSrc(getVideoUrl(savedJobId));
          setAppState("analyzing");
          startPolling(savedJobId);
        } else {
          localStorage.removeItem(STORAGE_KEY);
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY);
        setVideoSrc(null);
        setAppState("idle");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Find the closest frame for a given timestamp
  const findFrameAtTime = useCallback(
    (time: number): FrameData | null => {
      if (!analysisResult || analysisResult.frames.length === 0) return null;

      const frames = analysisResult.frames;
      let best = frames[0];
      let bestDiff = Math.abs(best.timestamp - time);

      for (let i = 1; i < frames.length; i++) {
        const diff = Math.abs(frames[i].timestamp - time);
        if (diff < bestDiff) {
          best = frames[i];
          bestDiff = diff;
        } else {
          // Frames are sorted by timestamp, so we can break early
          break;
        }
      }

      return best;
    },
    [analysisResult],
  );

  // Called by VideoPlayer on timeupdate
  const handleTimeUpdate = useCallback(
    (time: number) => {
      const frame = findFrameAtTime(time);
      setCurrentFrame(frame);
    },
    [findFrameAtTime],
  );

  // Start polling for analysis status
  const startPolling = useCallback((jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const status = await getAnalysisStatus(jobId);
        setProgress(status.progress);

        if (status.status === "completed") {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;

          const result = await getAnalysisResult(jobId);
          setAnalysisResult(result);
          setAppState("ready");
          // Set initial frame
          if (result.frames.length > 0) {
            setCurrentFrame(result.frames[0]);
          }
        } else if (status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setErrorMsg("Analysis failed. Please try again.");
          setAppState("error");
        }
      } catch (err) {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setErrorMsg(err instanceof ApiError ? err.detail : "Connection lost.");
        setAppState("error");
      }
    }, POLL_INTERVAL_MS);
  }, []);

  // Upload flow
  const handleVideoSelect = useCallback(
    async (file: File, previewUrl: string) => {
      setVideoSrc(previewUrl);
      setErrorMsg(null);
      setAppState("uploading");
      setProgress(0);

      try {
        const status = await uploadVideo(file);
        jobIdRef.current = status.jobId;
        localStorage.setItem(STORAGE_KEY, status.jobId);
        // Switch to backend-served video URL
        setVideoSrc(getVideoUrl(status.jobId));
        setAppState("analyzing");
        setProgress(status.progress);
        startPolling(status.jobId);
      } catch (err) {
        setErrorMsg(
          err instanceof ApiError ? err.detail : "Upload failed. Check your connection.",
        );
        setAppState("error");
      }
    },
    [startPolling],
  );

  // URL upload flow
  const handleUrlSubmit = useCallback(
    async (url: string) => {
      setVideoSrc(null);
      setErrorMsg(null);
      setAppState("uploading");
      setProgress(0);

      try {
        const status = await uploadVideoUrl(url);
        jobIdRef.current = status.jobId;
        localStorage.setItem(STORAGE_KEY, status.jobId);
        setVideoSrc(getVideoUrl(status.jobId));
        setAppState("analyzing");
        setProgress(status.progress);
        startPolling(status.jobId);
      } catch (err) {
        setErrorMsg(
          err instanceof ApiError ? err.detail : "Failed to download video. Check the URL.",
        );
        setAppState("error");
      }
    },
    [startPolling],
  );

  const handleReset = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    localStorage.removeItem(STORAGE_KEY);
    setVideoSrc(null);
    setAppState("idle");
    setProgress(0);
    setErrorMsg(null);
    setAnalysisResult(null);
    setCurrentFrame(null);
    jobIdRef.current = null;
    setIsCameraActive(false);
  }, []);

  // Camera mode handlers
  const handleModeChange = useCallback((mode: AppMode) => {
    if (mode === appMode) return;
    handleReset();
    setAppMode(mode);
    if (mode === "camera") {
      setIsCameraActive(true);
    }
  }, [appMode, handleReset]);

  const handleCameraFrameData = useCallback((frame: FrameData) => {
    setCurrentFrame(frame);
  }, []);

  const handleCameraStop = useCallback(() => {
    setIsCameraActive(false);
    setCurrentFrame(null);
    setAppMode("upload");
  }, []);

  // Determine what to show in the main area
  const renderMain = () => {
    // Camera mode
    if (appMode === "camera" && isCameraActive) {
      return (
        <CameraCapture
          onFrameData={handleCameraFrameData}
          onStop={handleCameraStop}
        />
      );
    }

    // Show uploader when idle or error-before-video
    if (!videoSrc) {
      return <VideoUploader onVideoSelect={handleVideoSelect} onUrlSubmit={handleUrlSubmit} />;
    }

    return (
      <div className="flex flex-1 flex-col gap-3 min-h-[200px] sm:min-h-[300px] lg:min-h-0">
        {/* Status banner */}
        {appState === "uploading" && (
          <StatusBanner
            label="Uploading video..."
            progress={null}
          />
        )}
        {appState === "analyzing" && (
          <StatusBanner
            label="Analyzing..."
            progress={progress}
          />
        )}
        {appState === "error" && (
          <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400 flex items-center justify-between">
            <span>{errorMsg}</span>
            <button
              type="button"
              onClick={handleReset}
              className="text-xs font-mono underline hover:text-red-300"
            >
              Try again
            </button>
          </div>
        )}

        <VideoPlayer
          src={videoSrc}
          onReset={handleReset}
          frameData={appState === "ready" ? currentFrame : null}
          onTimeUpdate={appState === "ready" ? handleTimeUpdate : undefined}
        />
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full min-h-screen">
      <Header
        hasVideo={!!videoSrc}
        onReset={handleReset}
        appMode={appMode}
        onModeChange={handleModeChange}
        isCameraActive={isCameraActive}
      />
      <main className="flex flex-1 flex-col lg:flex-row gap-2 sm:gap-4 p-2 sm:p-4">
        {renderMain()}
        <DashboardPanel
          frameData={
            isCameraActive || appState === "ready" ? currentFrame : undefined
          }
          allFrames={appState === "ready" ? analysisResult?.frames : undefined}
        />
      </main>
    </div>
  );
}

// ── Status Banner ────────────────────────────────────────────────────

function StatusBanner({
  label,
  progress,
}: {
  label: string;
  progress: number | null;
}) {
  const getStageLabel = (p: number) => {
    if (p < 10) return "Detecting holds...";
    if (p < 30) return "Building hold map...";
    if (p < 50) return "Detecting route...";
    if (p < 90) return "Analyzing frames...";
    return "Finishing up...";
  };

  const pct = progress ?? 0;

  return (
    <div className="rounded-lg bg-accent/10 border border-accent/30 px-4 py-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-accent animate-pulse" />
          <span className="text-sm text-accent font-medium">{label}</span>
        </div>
        <span className="text-xs font-mono text-accent">{Math.round(pct)}%</span>
      </div>
      {progress !== null && (
        <span className="text-xs text-muted">{getStageLabel(pct)}</span>
      )}
      <div className="h-1.5 rounded-full bg-border overflow-hidden">
        <div
          className="h-full rounded-full bg-accent transition-all duration-500"
          style={{ width: `${Math.max(pct, 2)}%` }}
        />
      </div>
    </div>
  );
}
