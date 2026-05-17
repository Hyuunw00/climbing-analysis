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
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
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
    <div className="flex flex-col gap-2 sm:gap-3 lg:flex-1 lg:min-h-0">
      {/* Video container with canvas overlay */}
      <div className="relative aspect-[3/4] sm:aspect-video lg:aspect-auto lg:flex-1 rounded-lg overflow-hidden bg-black">
        <video
          ref={videoRef}
          src={src}
          className="absolute inset-0 w-full h-full object-cover"
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

        {/* Volume control */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => {
              const video = videoRef.current;
              if (!video) return;
              video.muted = !video.muted;
              setIsMuted(video.muted);
            }}
            className="flex items-center justify-center w-11 h-11 sm:w-8 sm:h-8 rounded-md hover:bg-border transition-colors"
            aria-label={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted || volume === 0 ? (
              <svg className="w-4 h-4 text-muted" fill="currentColor" viewBox="0 0 24 24">
                <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.796 8.796 0 0021 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a8.99 8.99 0 003.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4 text-foreground" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
              </svg>
            )}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={isMuted ? 0 : volume}
            onChange={(e) => {
              const v = Number(e.target.value);
              const video = videoRef.current;
              if (!video) return;
              video.volume = v;
              setVolume(v);
              if (v > 0 && isMuted) {
                video.muted = false;
                setIsMuted(false);
              }
            }}
            className="w-16 sm:w-20 h-1 accent-accent cursor-pointer"
          />
        </div>

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
