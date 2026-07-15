"use client";

import { Loader2 } from "lucide-react";

interface EditProgressOverlayProps {
  show: boolean;
  progress: { done: number; total: number };
  status: "uploading" | "editing" | "done" | "error" | null;
}

/**
 * Non-blocking propagation pill. The edit is already visible on the current
 * frame (instant preview) — this just shows the background sweep filling in
 * the rest of the clip, without covering the editor.
 */
export function EditProgressOverlay({ show, progress, status }: EditProgressOverlayProps) {
  if (!show) return null;

  const percentage = progress.total > 0
    ? Math.round((progress.done / progress.total) * 100)
    : 0;

  return (
    <div
      className="fixed bottom-24 right-6 z-[9999] flex items-center gap-3 px-4 py-2.5 rounded-2xl border pointer-events-none"
      style={{
        background: "var(--ed-surface)",
        borderColor: "var(--ed-border)",
        boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
      }}
    >
      <Loader2 className="w-4 h-4 animate-spin shrink-0" style={{ color: "var(--accent)" }} />
      <div className="flex flex-col gap-1 min-w-[160px]">
        <div className="flex justify-between items-center gap-4">
          <span className="text-xs font-medium" style={{ color: "var(--ed-text)" }}>
            Propagating across clip
          </span>
          <span className="text-xs font-semibold tabular-nums" style={{ color: "var(--accent)" }}>
            {progress.total > 0 ? `${percentage}%` : "…"}
          </span>
        </div>
        <div
          className="w-full h-1 rounded-full overflow-hidden"
          style={{ background: "var(--ed-surface-2)" }}
        >
          <div
            className="h-full transition-all duration-300 ease-out rounded-full"
            style={{
              width: `${percentage}%`,
              background: "var(--accent)",
            }}
          />
        </div>
      </div>
    </div>
  );
}
