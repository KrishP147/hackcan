"use client";

import { Loader2, Sparkles } from "lucide-react";

interface EditProgressOverlayProps {
  show: boolean;
  progress: { done: number; total: number };
  status: "uploading" | "editing" | "done" | "error" | null;
}

export function EditProgressOverlay({ show, progress, status }: EditProgressOverlayProps) {
  if (!show) return null;

  const percentage = progress.total > 0 
    ? Math.round((progress.done / progress.total) * 100) 
    : 0;

  const statusText = status === "uploading" 
    ? "Uploading frames to Cloudinary..." 
    : status === "editing"
    ? "Applying edits..."
    : "Processing...";

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-[9999] pointer-events-none"
      style={{
        background: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(4px)",
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
            Applying Edit
          </h3>
          <p
            className="text-sm"
            style={{ color: "var(--ed-subtle)" }}
          >
            {statusText}
          </p>
        </div>

        {/* Progress Bar */}
        <div className="w-full">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-medium" style={{ color: "var(--ed-muted)" }}>
              {progress.done} / {progress.total} frames
            </span>
            <span className="text-xs font-semibold" style={{ color: "var(--accent)" }}>
              {percentage}%
            </span>
          </div>
          <div
            className="w-full h-2 rounded-full overflow-hidden"
            style={{ background: "var(--ed-surface-2)" }}
          >
            <div
              className="h-full transition-all duration-300 ease-out rounded-full"
              style={{
                width: `${percentage}%`,
                background: "var(--accent)",
                boxShadow: "0 0 8px rgba(244,63,94,0.5)",
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
