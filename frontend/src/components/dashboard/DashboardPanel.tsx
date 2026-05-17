"use client";

import type { FrameData } from "@/types";

// Dashboard metrics type
export interface DashboardData {
  routeColor: { color: string; label: string };
  verticalVelocity: number;
  performance: {
    smoothness: number;
    technique: number;
    stable: number;
    adhesion: number;
  };
  liveScore: number;
  avgScore: number;
}

// Default placeholder data (shown before analysis completes)
export const EMPTY_DASHBOARD_DATA: DashboardData = {
  routeColor: { color: "#666", label: "N/A" },
  verticalVelocity: 0,
  performance: {
    smoothness: 0,
    technique: 0,
    stable: 0,
    adhesion: 0,
  },
  liveScore: 0,
  avgScore: 0,
};

/**
 * Derive dashboard metrics from a single analysis frame.
 * These are approximations — a real scoring model would live on the backend.
 */
function deriveFromFrame(frame: FrameData): DashboardData {
  const holdsCount = frame.holds.length;
  const avgHoldConfidence =
    holdsCount > 0
      ? frame.holds.reduce((sum, h) => sum + h.confidence, 0) / holdsCount
      : 0;

  const poseCount = frame.pose?.length ?? 0;
  const avgPoseConfidence =
    poseCount > 0
      ? (frame.pose ?? []).reduce((sum, kp) => sum + kp.confidence, 0) / poseCount
      : 0;

  const topPrediction =
    frame.predictions.length > 0
      ? Math.max(...frame.predictions.map((p) => p.probability))
      : 0;

  const smoothness = Math.round(avgPoseConfidence * 100);
  const technique = Math.round(topPrediction * 100);
  const stable = Math.round(avgPoseConfidence * 80);
  const adhesion = Math.round(avgHoldConfidence * 100);
  const liveScore = Math.round((smoothness + technique + stable + adhesion) / 4);

  return {
    routeColor: { color: "#3b82f6", label: `${holdsCount} holds detected` },
    verticalVelocity: 0, // Requires multi-frame calculation
    performance: { smoothness, technique, stable, adhesion },
    liveScore,
    avgScore: liveScore,
  };
}

interface DashboardPanelProps {
  data?: DashboardData;
  frameData?: FrameData | null;
}

// Score grade based on value
function getGrade(score: number): string {
  if (score >= 90) return "A";
  if (score >= 75) return "B";
  if (score >= 60) return "C";
  return "D";
}

// Performance bar config
const PERF_BARS = [
  { key: "smoothness" as const, label: "Smoothness", color: "#00ff88" },
  { key: "technique" as const, label: "Technique", color: "#00aaff" },
  { key: "stable" as const, label: "Stable", color: "#ff8800" },
  { key: "adhesion" as const, label: "Adhesion", color: "#ff44aa" },
];

// Score range segments for bottom bar
const SCORE_SEGMENTS = [
  { min: 0, max: 25, color: "#ff4444", label: "D" },
  { min: 25, max: 50, color: "#ff8800", label: "C" },
  { min: 50, max: 75, color: "#00aaff", label: "B" },
  { min: 75, max: 100, color: "#00ff88", label: "A" },
];

export default function DashboardPanel({ data, frameData }: DashboardPanelProps) {
  // Priority: explicit data > derived from frameData > empty placeholder
  const resolvedData = data ?? (frameData ? deriveFromFrame(frameData) : EMPTY_DASHBOARD_DATA);
  const grade = getGrade(resolvedData.avgScore);

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border bg-panel p-4 lg:w-80 overflow-y-auto">
      <h2 className="text-sm font-bold tracking-widest uppercase text-accent">
        Climbing Analytics
      </h2>

      {/* Route Color */}
      <section className="flex flex-col gap-1">
        <span className="text-xs font-mono uppercase text-muted">Route Color</span>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-4 h-4 rounded-full border border-border"
            style={{ backgroundColor: resolvedData.routeColor.color }}
          />
          <span className="text-sm font-mono text-foreground">{resolvedData.routeColor.label}</span>
        </div>
      </section>

      {/* Vertical Velocity */}
      <section className="flex flex-col gap-1">
        <span className="text-xs font-mono uppercase text-muted">Vertical Velocity</span>
        <p className="text-2xl font-bold text-foreground">
          {resolvedData.verticalVelocity}
          <span className="text-sm font-normal text-muted ml-1">px/s</span>
        </p>
      </section>

      {/* Live Performance Bars */}
      <section className="flex flex-col gap-3">
        <span className="text-xs font-mono uppercase text-muted">Live Performance</span>
        {PERF_BARS.map(({ key, label, color }) => (
          <div key={key} className="flex flex-col gap-1">
            <div className="flex justify-between items-center">
              <span className="text-xs font-mono text-foreground">{label}</span>
              <span className="text-xs font-mono" style={{ color }}>
                {resolvedData.performance[key]}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-border overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${resolvedData.performance[key]}%`,
                  backgroundColor: color,
                }}
              />
            </div>
          </div>
        ))}
      </section>

      {/* Live Score */}
      <section className="mt-2 border-t border-border pt-4 flex flex-col gap-1">
        <span className="text-xs font-mono uppercase text-muted">Live Score</span>
        <p className="text-4xl font-bold" style={{ color: "#00ff88" }}>
          {resolvedData.liveScore}
          <span className="text-lg text-muted"> / 100</span>
        </p>
      </section>

      {/* Avg Score */}
      <section className="flex flex-col gap-1">
        <span className="text-xs font-mono uppercase text-muted">Avg. Score</span>
        <div className="flex items-baseline gap-2">
          <p className="text-3xl font-bold" style={{ color: "#00ff88" }}>
            {resolvedData.avgScore}
            <span className="text-lg text-muted"> / 100</span>
          </p>
          <span
            className="text-lg font-bold px-2 py-0.5 rounded"
            style={{
              backgroundColor: SCORE_SEGMENTS.find((s) => resolvedData.avgScore >= s.min && resolvedData.avgScore < s.max)?.color ?? "#00ff88",
              color: "#0a0a0a",
            }}
          >
            {grade}
          </span>
        </div>
      </section>

      {/* Score Range Bar */}
      <section className="flex flex-col gap-1">
        <div className="flex h-3 rounded-full overflow-hidden">
          {SCORE_SEGMENTS.map((seg) => (
            <div
              key={seg.label}
              className="flex-1 relative"
              style={{ backgroundColor: seg.color }}
            >
              <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-black/70">
                {seg.label}
              </span>
            </div>
          ))}
        </div>
        {/* Marker for avg score */}
        <div className="relative h-2">
          <div
            className="absolute top-0 w-0.5 h-2 bg-foreground"
            style={{ left: `${resolvedData.avgScore}%` }}
          />
        </div>
      </section>
    </div>
  );
}
