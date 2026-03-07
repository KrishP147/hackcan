"use client";

import { useCallback, useRef, useState } from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";

interface EditorTimelineProps {
  videoLoaded: boolean;
  currentFrame: number;
  totalFrames: number;
  duration: number;
  fps: number;
  isPlaying: boolean;
  isProcessing: boolean;
  zoom: number;
  editRangeStart: number;
  editRangeEnd: number;
  onFrameChange: (frame: number) => void;
  onTogglePlay: () => void;
  onZoomChange: (zoom: number) => void;
  onEditRangeChange: (start: number, end: number) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function EditorTimeline({
  videoLoaded,
  currentFrame,
  totalFrames,
  duration,
  fps,
  isPlaying,
  isProcessing,
  zoom,
  editRangeStart,
  editRangeEnd,
  onFrameChange,
  onTogglePlay,
  onZoomChange,
  onEditRangeChange,
}: EditorTimelineProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const rangeRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<"start" | "end" | "bar" | null>(null);
  const dragStartX = useRef(0);
  const dragStartRange = useRef({ start: 0, end: 0 });

  const currentTime = totalFrames > 0 ? currentFrame / fps : 0;
  const progress = totalFrames > 0 ? (currentFrame / totalFrames) * 100 : 0;

  const handleTrackClick = useCallback(
    (e: React.MouseEvent) => {
      if (!trackRef.current || !videoLoaded) return;
      const rect = trackRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      onFrameChange(Math.round(pct * (totalFrames - 1)));
    },
    [totalFrames, videoLoaded, onFrameChange]
  );

  const maxFrame = Math.max(totalFrames - 1, 1);
  const rangeStartPct = totalFrames > 0 ? (editRangeStart / maxFrame) * 100 : 0;
  const rangeEndPct = totalFrames > 0 ? (editRangeEnd / maxFrame) * 100 : 100;

  const handleRangePointerDown = useCallback(
    (e: React.PointerEvent, handle: "start" | "end" | "bar") => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(handle);
      dragStartX.current = e.clientX;
      dragStartRange.current = { start: editRangeStart, end: editRangeEnd };

      const onMove = (ev: PointerEvent) => {
        if (!rangeRef.current) return;
        const rect = rangeRef.current.getBoundingClientRect();
        const deltaFrames = Math.round(((ev.clientX - dragStartX.current) / rect.width) * maxFrame);

        if (handle === "start") {
          const newStart = Math.max(0, Math.min(dragStartRange.current.end - 1, dragStartRange.current.start + deltaFrames));
          onEditRangeChange(newStart, dragStartRange.current.end);
        } else if (handle === "end") {
          const newEnd = Math.max(dragStartRange.current.start + 1, Math.min(maxFrame, dragStartRange.current.end + deltaFrames));
          onEditRangeChange(dragStartRange.current.start, newEnd);
        } else {
          const rangeLen = dragStartRange.current.end - dragStartRange.current.start;
          let newStart = dragStartRange.current.start + deltaFrames;
          newStart = Math.max(0, Math.min(maxFrame - rangeLen, newStart));
          onEditRangeChange(newStart, newStart + rangeLen);
        }
      };

      const onUp = () => {
        setDragging(null);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [editRangeStart, editRangeEnd, maxFrame, onEditRangeChange]
  );

  return (
    <div
      className="h-[174px] border-t flex flex-col shrink-0"
      style={{ background: "var(--ed-surface)", borderColor: "var(--ed-border)" }}
    >
      {/* Playback controls */}
      <div
        className="flex items-center justify-center gap-4 py-2.5 border-b"
        style={{ borderColor: "var(--ed-surface-2)" }}
      >
        <span className="text-xs font-mono w-10 text-right tabular-nums" style={{ color: "var(--ed-subtle)" }}>
          {formatTime(currentTime)}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            disabled={!videoLoaded}
            onClick={() => onFrameChange(Math.max(0, currentFrame - 1))}
            className="w-7 h-7 flex items-center justify-center rounded-lg transition-all"
            style={{ color: "var(--ed-icon)", opacity: !videoLoaded ? 0.25 : 1 }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--ed-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>
          <button
            disabled={!videoLoaded}
            onClick={onTogglePlay}
            className="w-9 h-9 flex items-center justify-center rounded-full transition-all hover:scale-105"
            style={{
              background: "var(--ed-text)",
              color: "var(--ed-surface)",
              opacity: !videoLoaded ? 0.25 : 1,
            }}
          >
            {isPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4 ml-0.5" fill="currentColor" />
            )}
          </button>
          <button
            disabled={!videoLoaded}
            onClick={() => onFrameChange(Math.min(totalFrames - 1, currentFrame + 1))}
            className="w-7 h-7 flex items-center justify-center rounded-lg transition-all"
            style={{ color: "var(--ed-icon)", opacity: !videoLoaded ? 0.25 : 1 }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--ed-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>
        </div>
        <span className="text-xs font-mono w-10 tabular-nums" style={{ color: "var(--ed-subtle)" }}>
          {formatTime(duration)}
        </span>
      </div>

      {/* Timeline track area */}
      <div className="flex-1 flex flex-col px-4 py-2.5 gap-2">
        <div className="relative">
          {/* Time markers */}
          <div className="flex justify-between text-[9px] font-mono mb-1.5 px-0.5" style={{ color: "var(--ed-disabled)" }}>
            {Array.from({ length: 6 }, (_, i) => (
              <span key={i}>{formatTime((duration / 5) * i)}</span>
            ))}
          </div>

          {/* Edit range selector */}
          {videoLoaded && (
            <div
              ref={rangeRef}
              className="h-5 rounded-lg relative mb-1.5 cursor-default"
              style={{ background: "var(--ed-surface-2)" }}
            >
              {/* Highlighted range bar */}
              <div
                className="absolute top-0 h-full rounded-md transition-colors"
                style={{
                  left: `${rangeStartPct}%`,
                  width: `${rangeEndPct - rangeStartPct}%`,
                  background: dragging === "bar" ? "rgba(244,63,94,0.25)" : "rgba(244,63,94,0.15)",
                  cursor: "grab",
                }}
                onPointerDown={(e) => handleRangePointerDown(e, "bar")}
              />
              {/* Start handle */}
              <div
                className="absolute top-0 h-full w-2 rounded-l-md z-10 group"
                style={{
                  left: `calc(${rangeStartPct}% - 4px)`,
                  cursor: "ew-resize",
                }}
                onPointerDown={(e) => handleRangePointerDown(e, "start")}
              >
                <div
                  className="w-full h-full rounded-l-md transition-colors"
                  style={{ background: dragging === "start" ? "var(--accent)" : "rgba(244,63,94,0.6)" }}
                />
              </div>
              {/* End handle */}
              <div
                className="absolute top-0 h-full w-2 rounded-r-md z-10 group"
                style={{
                  left: `${rangeEndPct}%`,
                  cursor: "ew-resize",
                }}
                onPointerDown={(e) => handleRangePointerDown(e, "end")}
              >
                <div
                  className="w-full h-full rounded-r-md transition-colors"
                  style={{ background: dragging === "end" ? "var(--accent)" : "rgba(244,63,94,0.6)" }}
                />
              </div>
              {/* Range label */}
              <div
                className="absolute top-0 h-full flex items-center justify-center pointer-events-none"
                style={{
                  left: `${rangeStartPct}%`,
                  width: `${rangeEndPct - rangeStartPct}%`,
                }}
              >
                {(rangeEndPct - rangeStartPct) > 10 && (
                  <span className="text-[9px] font-mono font-medium" style={{ color: "var(--accent)" }}>
                    {formatTime(editRangeStart / fps)} – {formatTime(editRangeEnd / fps)}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Scrubber track */}
          <div
            ref={trackRef}
            onClick={handleTrackClick}
            className="h-8 rounded-xl cursor-pointer relative overflow-hidden border"
            style={{ background: "var(--ed-surface-2)", borderColor: "var(--ed-border)" }}
          >
            {videoLoaded && (
              <div className="absolute inset-0 flex">
                {Array.from({ length: 20 }, (_, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    style={{ borderRight: "1px solid var(--ed-border)", opacity: 0.4 }}
                  />
                ))}
              </div>
            )}
            {/* Edit range highlight on scrubber */}
            {videoLoaded && (
              <div
                className="absolute top-0 h-full pointer-events-none"
                style={{
                  left: `${rangeStartPct}%`,
                  width: `${rangeEndPct - rangeStartPct}%`,
                  background: "rgba(244,63,94,0.08)",
                  borderLeft: "1px solid rgba(244,63,94,0.3)",
                  borderRight: "1px solid rgba(244,63,94,0.3)",
                }}
              />
            )}
            {videoLoaded && (
              <div
                className="absolute top-0 left-0 h-full transition-all duration-75"
                style={{ width: `${progress}%`, background: "rgba(244,63,94,0.12)" }}
              />
            )}
            {videoLoaded && (
              <div
                className="absolute top-0 h-full w-0.5 bg-[var(--accent)] transition-all duration-75 z-10"
                style={{ left: `${progress}%` }}
              >
                <div className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-2 h-2 bg-[var(--accent)] rotate-45 rounded-sm" />
              </div>
            )}
            {isProcessing && (
              <div className="absolute inset-0" style={{ background: "rgba(244,63,94,0.08)" }}>
                <div className="h-full animate-progress-bar" style={{ background: "rgba(244,63,94,0.20)" }} />
              </div>
            )}
          </div>
        </div>

        {/* Bottom row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs" style={{ color: "var(--ed-disabled)" }}>
            <span className="text-base leading-none">+</span>
            <span>or drag and drop media</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="25"
              max="200"
              value={zoom}
              onChange={(e) => onZoomChange(Number(e.target.value))}
              className="w-20 accent-[var(--accent)]"
            />
            <span className="text-[10px] font-mono w-8 tabular-nums" style={{ color: "var(--ed-subtle)" }}>
              {zoom}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
