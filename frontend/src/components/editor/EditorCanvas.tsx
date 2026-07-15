"use client";

import { useCallback, useRef } from "react";
import type { Detection, EditMode, EditParams } from "@/lib/mock-data";
import { BoundingBox } from "./BoundingBox";
import type { EditAction } from "./EditToolbar";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface EditorCanvasProps {
  projectId: string | null;
  videoLoaded: boolean;
  detections: Detection[];
  isDetecting: boolean;
  isSegmenting: boolean;
  segmentStatus: string | null;
  segmentAnchorFrame: number | null;
  maskCount: number;
  maskVersion: number;
  editVersion: number;
  transformedFrameVersions?: { [frameIndex: number]: number };
  selectedObjectId: string | null;
  editMode: EditMode | null;
  editParams: EditParams;
  isProcessing: boolean;
  zoom: number;
  currentFrame: number;
  totalFrames: number;
  frameWidth: number;
  frameHeight: number;
  previewFrameUrl: string | null;
  instantPreviewUrl?: string | null;
  instantPreviewFrame?: number | null;
  pendingEditAction?: string | null;
  isEditPreviewing?: boolean;
  aiEditStatus: "idle" | "preview" | "applying" | "done";
  storageBaseUrl: string | null;
  onSelectObject: (id: string | null) => void;
  onUpload: () => void;
  onApplyEdit: (action: EditAction, params: { color?: string; prompt?: string; scale?: number }) => void;
  onSegmentAtPoint: (clickX: number, clickY: number) => void;
  onConfirmPropagation: () => void;
  onConfirmEditPropagation: () => void;
  onCancelEditPreview: () => void;
  onCancelEdit: () => void;
}

const EDIT_LABELS: Record<string, string> = {
  delete: "Remove",
  recolor: "Recolor",
  resize: "Resize",
  blur_region: "Blur",
  move: "Move",
  color_pop: "Color Pop",
  glow: "Glow",
  replace: "Replace",
  bg_replace: "Replace Background",
};

export function EditorCanvas({
  projectId,
  videoLoaded,
  detections,
  isDetecting,
  isSegmenting,
  segmentStatus,
  segmentAnchorFrame,
  maskCount,
  maskVersion,
  editVersion,
  transformedFrameVersions,
  selectedObjectId,
  editMode,
  editParams,
  isProcessing,
  zoom,
  currentFrame,
  totalFrames,
  frameWidth,
  frameHeight,
  previewFrameUrl,
  instantPreviewUrl,
  instantPreviewFrame,
  pendingEditAction,
  isEditPreviewing,
  aiEditStatus,
  storageBaseUrl,
  onSelectObject,
  onUpload,
  onApplyEdit,
  onSegmentAtPoint,
  onConfirmPropagation,
  onConfirmEditPropagation,
  onCancelEditPreview,
  onCancelEdit,
}: EditorCanvasProps) {
  const imgRef = useRef<HTMLDivElement>(null);
  const hasMaskForCurrentFrame = maskCount > 0 && (
    segmentStatus === "done" || segmentAnchorFrame === currentFrame + 1
  );

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (isProcessing || isSegmenting || !imgRef.current || !frameWidth || !frameHeight) {
        return;
      }
      e.stopPropagation();
      const rect = imgRef.current.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width;
      const relY = (e.clientY - rect.top) / rect.height;
      // Only segment if click is within the frame bounds
      if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;
      const clickX = Math.round(relX * frameWidth);
      const clickY = Math.round(relY * frameHeight);
      onSegmentAtPoint(clickX, clickY);
    },
    [frameWidth, frameHeight, isProcessing, isSegmenting, onSegmentAtPoint]
  );

  if (!videoLoaded) {
    return <EmptyCanvas onUpload={onUpload} />;
  }

  // Show preview frame if in preview mode, otherwise show current frame
  // Use per-frame versioning for transformed frames, otherwise use global editVersion
  const currentFrameIndex = currentFrame + 1; // Backend uses 1-based indexing
  const frameVersion = transformedFrameVersions?.[currentFrameIndex] ?? editVersion;
  const paddedIndex = String(currentFrameIndex).padStart(4, "0");
  // Instant preview wins for its own frame — the edit appears the moment the
  // button is pressed, then the propagated real frame replaces it
  const frameUrl = instantPreviewUrl != null && instantPreviewFrame === currentFrame
    ? instantPreviewUrl
    : aiEditStatus === "preview" && previewFrameUrl
    ? previewFrameUrl
    : projectId
    ? storageBaseUrl && frameVersion === 0
      ? `${storageBaseUrl}/frame_${paddedIndex}.jpg`
      : `${API_URL}/frame/${projectId}/${currentFrameIndex}?v=${frameVersion}`
    : null;

  return (
    <div
      className="flex-1 flex items-center justify-center overflow-hidden relative"
      style={{ background: "var(--ed-bg)" }}
    >
      {isDetecting && (
        <div className="absolute inset-0 z-20 pointer-events-none animate-detection-shimmer" />
      )}

      <div
        className="relative"
        style={{ transform: `scale(${zoom / 100})`, transition: "transform 200ms ease" }}
      >
        <div
          ref={imgRef}
          className="w-[768px] h-[432px] rounded-2xl overflow-hidden relative shadow-2xl cursor-crosshair"
          style={{
            background: "var(--ed-surface-2)",
            boxShadow: "0 25px 60px rgba(0,0,0,0.25)",
          }}
          onClick={handleCanvasClick}
        >
          {!isSegmenting && !hasMaskForCurrentFrame && aiEditStatus !== "preview" && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 pointer-events-none rounded-xl border px-4 py-2 text-center backdrop-blur-md"
              style={{
                background: "rgba(10, 10, 10, 0.72)",
                borderColor: "rgba(255,255,255,0.14)",
              }}
            >
              <p className="text-xs font-semibold text-white">Click an object to start</p>
              <p className="mt-0.5 text-[10px] text-white/60">SAM 2 will isolate your selection</p>
            </div>
          )}

          {frameUrl ? (
            <>
              <img
                src={frameUrl}
                alt={aiEditStatus === "preview" ? "AI Preview" : `Frame ${currentFrame + 1}`}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              />
              {aiEditStatus === "preview" && (
                <div className="absolute top-4 left-4 z-30 px-3 py-1.5 rounded-xl text-xs font-medium border"
                  style={{
                    background: "rgba(244,63,94,0.9)",
                    color: "#fff",
                    borderColor: "var(--accent)",
                  }}
                >
                  AI Preview
                </div>
              )}
            </>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-sm font-medium" style={{ color: "var(--ed-disabled)" }}>
                Frame Preview
              </div>
            </div>
          )}


          {/* Hide masks and detections when showing AI preview */}
          {aiEditStatus !== "preview" && projectId && hasMaskForCurrentFrame && !isSegmenting && (
            <>
              <img
                src={`${API_URL}/mask-outline/${projectId}/${currentFrameIndex}?v=${maskVersion}`}
                alt=""
                className="absolute inset-0 w-full h-full object-contain pointer-events-none z-[2]"
                style={{ filter: "drop-shadow(0 0 3px rgba(244,63,94,0.9))" }}
              />

            </>
          )}

          {isSegmenting && (
            <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
              <div
                className="flex flex-col items-center gap-2 px-5 py-3.5 rounded-2xl border"
                style={{ background: "rgba(0,0,0,0.7)", borderColor: "rgba(255,255,255,0.1)" }}
              >
                <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                <span className="text-white/70 text-xs font-medium">
                  {segmentStatus === "propagating" ? "Tracking object through video…" : "Segmenting keyframe…"}
                </span>
              </div>
            </div>
          )}

          {isEditPreviewing && (
            <div className="absolute inset-0 flex items-center justify-center z-30 pointer-events-none">
              <div
                className="flex flex-col items-center gap-2 px-5 py-3.5 rounded-2xl border"
                style={{ background: "rgba(0,0,0,0.76)", borderColor: "rgba(255,255,255,0.12)" }}
              >
                <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                <span className="text-white/75 text-xs font-medium">
                  Preparing {pendingEditAction ? EDIT_LABELS[pendingEditAction] || pendingEditAction : "edit"} preview…
                </span>
                <span className="text-white/40 text-[10px]">Current frame only</span>
              </div>
            </div>
          )}

          {segmentStatus === "keyframe_ready" && !isSegmenting && (
            <div
              className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 w-[360px] rounded-2xl border p-4 shadow-2xl backdrop-blur-xl"
              style={{ background: "rgba(15,15,18,0.92)", borderColor: "rgba(255,255,255,0.14)" }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-400">
                  ✓
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-white">Keyframe selection ready</p>
                  <p className="mt-1 text-xs leading-5 text-white/60">
                    Track this object through all {totalFrames} frames?
                  </p>
                  <button
                    type="button"
                    onClick={onConfirmPropagation}
                    className="mt-3 w-full rounded-xl bg-[var(--accent)] px-4 py-2.5 text-xs font-semibold text-white transition hover:brightness-110"
                  >
                    Segment all frames
                  </button>
                  <p className="mt-2 text-center text-[10px] text-white/40">
                    Or click another point to adjust the selection
                  </p>
                </div>
              </div>
            </div>
          )}

          {pendingEditAction && !isEditPreviewing && instantPreviewUrl && (
            <div
              className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 w-[380px] rounded-2xl border p-4 shadow-2xl backdrop-blur-xl"
              style={{ background: "rgba(15,15,18,0.94)", borderColor: "rgba(255,255,255,0.14)" }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-400">
                  ✓
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-white">
                    {EDIT_LABELS[pendingEditAction] || pendingEditAction} preview ready
                  </p>
                  <p className="mt-1 text-xs leading-5 text-white/60">
                    Apply this change to all {totalFrames} frames?
                  </p>
                  <div className="mt-3 flex gap-2">
                    <button
                      type="button"
                      onClick={onCancelEditPreview}
                      className="flex-1 rounded-xl border border-white/15 px-4 py-2.5 text-xs font-semibold text-white/70 transition hover:bg-white/5"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={onConfirmEditPropagation}
                      className="flex-[1.5] rounded-xl bg-[var(--accent)] px-4 py-2.5 text-xs font-semibold text-white transition hover:brightness-110"
                    >
                      Apply to all frames
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {aiEditStatus !== "preview" && detections.map((det) => (
            <BoundingBox
              key={det.id}
              detection={det}
              isSelected={selectedObjectId === det.id}
              onClick={() => onSelectObject(det.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function EmptyCanvas({ onUpload }: { onUpload: () => void }) {
  return (
    <div
      className="flex-1 flex items-center justify-center"
      style={{ background: "var(--ed-bg)" }}
    >
      <div
        className="flex flex-col items-center gap-4 p-14 rounded-2xl border-2 border-dashed cursor-pointer group transition-all"
        style={{ borderColor: "var(--ed-border)" }}
        onClick={onUpload}
        onMouseEnter={(e) => (e.currentTarget.style.borderColor = "rgba(244,63,94,0.5)")}
        onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--ed-border)")}
      >
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center transition-colors"
          style={{ background: "var(--ed-overlay)" }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--ed-icon)" }}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium" style={{ color: "var(--ed-muted)" }}>
            Upload a video to start editing
          </p>
          <p className="text-xs mt-1" style={{ color: "var(--ed-subtle)" }}>
            Drag and drop or click to browse
          </p>
        </div>
      </div>
    </div>
  );
}
