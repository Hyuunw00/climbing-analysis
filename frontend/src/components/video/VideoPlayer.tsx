"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FrameData } from "@/types";
import OverlayRenderer from "./OverlayRenderer";

interface VideoPlayerProps {
  src: string;
  onReset?: () => void;
  frameData?: FrameData | null;
  onTimeUpdate?: (time: number) => void;
}

export default function VideoPlayer({ src, onReset, frameData = null, onTimeUpdate }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
    } else {
      video.pause();
    }
  }, []);

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Number(e.target.value);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      onTimeUpdate?.(video.currentTime);
    };
    const onLoadedMetadata = () => setDuration(video.duration);

    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("loadedmetadata", onLoadedMetadata);

    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
    };
  }, [onTimeUpdate]);

  return (
    <div className="flex flex-1 flex-col gap-3 min-h-[200px] sm:min-h-[300px] lg:min-h-0">
      {/* Video container with canvas overlay */}
      <div className="relative flex-1 rounded-lg overflow-hidden bg-black">
        <video
          ref={videoRef}
          src={src}
          className="absolute inset-0 w-full h-full object-contain"
          playsInline
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
          isPlaying={isPlaying}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 rounded-lg bg-panel border border-border px-3 py-2">
        <button
          type="button"
          onClick={togglePlay}
          className="flex items-center justify-center w-11 h-11 sm:w-8 sm:h-8 rounded-md hover:bg-border transition-colors"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg className="w-4 h-4 text-foreground" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4 text-foreground" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        {/* Time display */}
        <span className="text-sm sm:text-xs font-mono text-muted min-w-[70px]">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        {/* Seek bar */}
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.1}
          value={currentTime}
          onChange={handleSeek}
          className="flex-1 h-1 accent-accent cursor-pointer"
        />

        {/* Upload another */}
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="text-xs text-muted hover:text-foreground transition-colors font-mono"
          >
            New
          </button>
        )}
      </div>
    </div>
  );
}
