import type { AppMode } from "@/types";

interface HeaderProps {
  hasVideo?: boolean;
  onReset?: () => void;
  appMode: AppMode;
  onModeChange: (mode: AppMode) => void;
  isCameraActive?: boolean;
}

export default function Header({ hasVideo, onReset, appMode, onModeChange, isCameraActive }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-border">
      <h1 className="text-sm font-bold tracking-widest uppercase">
        Climbing Analytics
      </h1>
      <div className="flex items-center gap-3">
        {/* Mode toggle - show when idle (no video and not in camera mode) */}
        {!hasVideo && !isCameraActive && (
          <div className="flex items-center rounded border border-border overflow-hidden">
            <button
              type="button"
              onClick={() => onModeChange("upload")}
              className={`text-xs font-mono px-2.5 py-1 transition-colors ${
                appMode === "upload"
                  ? "bg-accent text-black"
                  : "text-muted hover:text-foreground"
              }`}
            >
              Upload
            </button>
            <button
              type="button"
              onClick={() => onModeChange("camera")}
              className={`text-xs font-mono px-2.5 py-1 transition-colors ${
                appMode === "camera"
                  ? "bg-accent text-black"
                  : "text-muted hover:text-foreground"
              }`}
            >
              Camera
            </button>
          </div>
        )}

        {hasVideo && onReset && (
          <button
            type="button"
            onClick={onReset}
            className="text-xs font-mono px-2 py-1 rounded border border-border hover:bg-panel transition-colors"
          >
            New Video
          </button>
        )}
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rec opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-rec" />
          </span>
          <span className="text-xs font-mono text-rec">REC</span>
        </div>
      </div>
    </header>
  );
}
