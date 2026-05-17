"use client";

import { useCallback, useRef, useState } from "react";

interface VideoUploaderProps {
  onVideoSelect: (file: File, previewUrl: string) => void;
}

const ACCEPTED_FORMATS = ["video/mp4", "video/quicktime", "video/webm"];
const MAX_SIZE_MB = 100;
const MAX_DURATION_SEC = 120;

export default function VideoUploader({ onVideoSelect }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback(
    (file: File): Promise<string | null> => {
      return new Promise((resolve) => {
        // Check format
        if (!ACCEPTED_FORMATS.includes(file.type)) {
          resolve("Unsupported format. Please upload mp4, mov, or webm.");
          return;
        }

        // Check size
        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
          resolve(`File too large. Maximum size is ${MAX_SIZE_MB}MB.`);
          return;
        }

        // Check duration
        const video = document.createElement("video");
        video.preload = "metadata";
        video.onloadedmetadata = () => {
          URL.revokeObjectURL(video.src);
          if (video.duration > MAX_DURATION_SEC) {
            resolve(`Video too long. Maximum duration is ${MAX_DURATION_SEC / 60} minutes.`);
          } else {
            resolve(null);
          }
        };
        video.onerror = () => {
          URL.revokeObjectURL(video.src);
          resolve("Could not read video file.");
        };
        video.src = URL.createObjectURL(file);
      });
    },
    []
  );

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setProgress(0);

      const validationError = await validateFile(file);
      if (validationError) {
        setError(validationError);
        setProgress(null);
        return;
      }

      // Validation passed — hand off to parent for actual upload
      const previewUrl = URL.createObjectURL(file);
      setProgress(null);
      onVideoSelect(file, previewUrl);
    },
    [validateFile, onVideoSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`flex flex-1 flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed min-h-[300px] lg:min-h-0 transition-colors cursor-pointer ${
        isDragging
          ? "border-accent bg-accent/5"
          : "border-border bg-panel hover:border-muted"
      }`}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload video"
    >
      {/* Upload icon */}
      <svg
        className="w-12 h-12 text-muted"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
        />
      </svg>

      <div className="flex flex-col items-center gap-1 text-center px-4">
        <p className="text-sm text-foreground font-medium">
          Drop your climbing video here
        </p>
        <p className="text-xs text-muted">
          MP4, MOV, or WebM — max 2 min, 100MB
        </p>
      </div>

      <button
        type="button"
        className="mt-2 px-4 py-2 rounded-md bg-accent text-background text-sm font-medium hover:bg-accent/90 transition-colors"
        onClick={(e) => {
          e.stopPropagation();
          inputRef.current?.click();
        }}
      >
        Select File
      </button>

      {/* Progress bar */}
      {progress !== null && (
        <div className="w-full max-w-xs">
          <div className="h-1.5 rounded-full bg-border overflow-hidden">
            <div
              className="h-full rounded-full bg-accent transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-muted text-center mt-1">
            {Math.round(progress)}% uploaded
          </p>
        </div>
      )}

      {/* Error message */}
      {error && (
        <p className="text-xs text-rec px-4 text-center">{error}</p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/webm"
        className="hidden"
        onChange={handleInputChange}
      />
    </div>
  );
}
