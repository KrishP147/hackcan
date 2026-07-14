"use client";

import { Film, Loader2 } from "lucide-react";

interface ExportProgressOverlayProps {
  show: boolean;
  videoName: string;
}

export function ExportProgressOverlay({ show, videoName }: ExportProgressOverlayProps) {
  if (!show) return null;

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-progress-title"
      aria-describedby="export-progress-description"
      style={{
        background: "rgba(0, 0, 0, 0.78)",
        backdropFilter: "blur(8px)",
      }}
    >
      <div
        className="w-full max-w-md rounded-3xl p-8 sm:p-10"
        style={{
          background: "var(--ed-surface)",
          border: "1px solid var(--ed-border)",
          boxShadow: "0 24px 80px rgba(0, 0, 0, 0.4)",
        }}
      >
        <div className="flex flex-col items-center text-center">
          <div
            className="relative mb-6 flex h-16 w-16 items-center justify-center rounded-2xl"
            style={{ background: "rgba(244, 63, 94, 0.14)" }}
          >
            <Film className="h-7 w-7" style={{ color: "var(--accent)" }} strokeWidth={1.6} />
            <span
              className="absolute -right-1 -top-1 flex h-6 w-6 items-center justify-center rounded-full"
              style={{ background: "var(--ed-surface)", border: "1px solid var(--ed-border)" }}
            >
              <Loader2 className="h-3.5 w-3.5 animate-spin" style={{ color: "var(--accent)" }} />
            </span>
          </div>

          <h2
            id="export-progress-title"
            className="text-xl font-semibold tracking-tight"
            style={{ color: "var(--ed-text)" }}
          >
            Exporting your video
          </h2>
          <p
            id="export-progress-description"
            className="mt-2 max-w-xs text-sm leading-6"
            style={{ color: "var(--ed-subtle)" }}
          >
            Rendering {videoName || "your project"} into an MP4. This can take a moment.
          </p>

          <div
            className="mt-7 h-2 w-full overflow-hidden rounded-full"
            aria-label="Export in progress"
            role="progressbar"
            aria-valuetext="Export in progress"
            style={{ background: "var(--ed-surface-2)" }}
          >
            <div
              className="h-full w-2/5 rounded-full animate-export-progress"
              style={{
                background: "var(--accent)",
                boxShadow: "0 0 14px rgba(244, 63, 94, 0.6)",
              }}
            />
          </div>

          <p className="mt-3 text-xs" style={{ color: "var(--ed-disabled)" }}>
            Keep this window open while the export finishes
          </p>
        </div>
      </div>
    </div>
  );
}
