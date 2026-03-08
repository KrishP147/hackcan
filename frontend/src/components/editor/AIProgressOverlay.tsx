"use client";

import { Sparkles, Loader2 } from "lucide-react";

interface AIProgressOverlayProps {
  show: boolean;
  progress: { done: number; total: number };
  interpolationProgress: { done: number; total: number };
  phase: "transforming" | "interpolating" | "done" | null;
  status: string;
}

export function AIProgressOverlay({ show, progress, interpolationProgress, phase, status }: AIProgressOverlayProps) {
  // Determine which progress to show based on phase
  const isTransforming = phase === "transforming" || phase === null;
  const isInterpolating = phase === "interpolating";
  const currentProgress = isInterpolating ? interpolationProgress : progress;
  
  // Calculate percentage - show 0% if total is 0 (initializing)
  const percentage = currentProgress.total > 0 
    ? Math.round((currentProgress.done / currentProgress.total) * 100) 
    : 0;
  
  // Show "Initializing..." if total is 0
  const isInitializing = currentProgress.total === 0;

  // Debug logging
  if (show) {
    console.log("[AIProgressOverlay] Showing overlay:", { status, phase, progress, interpolationProgress, percentage });
  }

  if (!show) return null;

  return (
    <div
      className="fixed inset-0 flex items-center justify-center"
      style={{
        background: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(4px)",
        zIndex: 9999,
      }}
    >
      <div
        className="flex flex-col items-center gap-6 p-8 rounded-2xl max-w-md w-full mx-4"
        style={{
          background: "var(--ed-surface)",
          border: "1px solid var(--ed-border)",
          boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.3)",
        }}
      >
        {/* Icon */}
        <div className="relative">
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{
              background: "var(--accent)",
              opacity: 0.1,
            }}
          >
            <Sparkles className="w-8 h-8" style={{ color: "var(--accent)" }} />
          </div>
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 
              className="w-6 h-6 animate-spin" 
              style={{ color: "var(--accent)" }} 
            />
          </div>
        </div>

        {/* Title */}
        <div className="text-center">
          <h3
            className="text-lg font-semibold mb-1"
            style={{ color: "var(--ed-text)" }}
          >
            {isInterpolating ? "Smoothing Video" : "Applying AI Edit"}
          </h3>
          <p
            className="text-sm"
            style={{ color: "var(--ed-subtle)" }}
          >
            {isInterpolating 
              ? "Interpolating frames with RIFE for smooth motion..." 
              : "Transforming key frames with AI..."}
          </p>
        </div>

        {/* Progress Bar */}
        <div className="w-full">
          <div className="flex justify-between items-center mb-2">
            <span
              className="text-xs font-medium"
              style={{ color: "var(--ed-text)" }}
            >
              {isInitializing 
                ? "Initializing..." 
                : isInterpolating
                  ? "Interpolating frames"
                  : status === "applying" 
                    ? "Transforming frames" 
                    : "Complete"}
            </span>
            <span
              className="text-xs font-semibold"
              style={{ color: "var(--accent)" }}
            >
              {percentage}%
            </span>
          </div>
          
          {/* Progress Bar Container */}
          <div
            className="w-full h-2 rounded-full overflow-hidden"
            style={{
              background: "var(--ed-surface-2)",
            }}
          >
            {/* Progress Fill */}
            <div
              className="h-full rounded-full transition-all duration-300 ease-out"
              style={{
                width: `${percentage}%`,
                background: "var(--accent)",
                boxShadow: "0 0 10px rgba(var(--accent-rgb), 0.5)",
              }}
            />
          </div>

          {/* Progress Text */}
          <div className="flex justify-between items-center mt-2">
            <span
              className="text-xs"
              style={{ color: "var(--ed-subtle)" }}
            >
              {isInitializing 
                ? isInterpolating
                  ? "Preparing interpolation..."
                  : "Preparing transformation..."
                : isInterpolating
                  ? `${interpolationProgress.done} of ${interpolationProgress.total} frames interpolated`
                  : `${progress.done} of ${progress.total} frames transformed`}
            </span>
            {status === "applying" && !isInitializing && (
              <span
                className="text-xs"
                style={{ color: "var(--ed-subtle)" }}
              >
                {isInterpolating
                  ? interpolationProgress.total - interpolationProgress.done
                  : progress.total - progress.done} remaining
              </span>
            )}
          </div>
        </div>

        {/* Status Message */}
        {status === "applying" && (
          <div className="flex items-center gap-2 mt-2">
            <div
              className="w-2 h-2 rounded-full animate-pulse"
              style={{ background: "var(--accent)" }}
            />
            <span
              className="text-xs"
              style={{ color: "var(--ed-subtle)" }}
            >
              This may take a few minutes...
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
