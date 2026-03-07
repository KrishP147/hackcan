"use client";

import { useCallback, useRef, useState } from "react";
import { Crosshair } from "lucide-react";
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
  maskCount: number;
  maskVersion: number;
  editVersion: number;
  selectedObjectId: string | null;
  editMode: EditMode | null;
  editParams: EditParams;
  isProcessing: boolean;
  zoom: number;
  currentFrame: number;
  totalFrames: number;
  frameWidth: number;
  frameHeight: number;
  onSelectObject: (id: string | null) => void;
  onUpload: () => void;
  onApplyEdit: (action: EditAction, params: { color?: string; prompt?: string; scale?: number }) => void;
  onSegmentAtPoint: (clickX: number, clickY: number) => void;
}

export function EditorCanvas({
  projectId,
  videoLoaded,
  detections,
  isDetecting,
  isSegmenting,
  maskCount,
  maskVersion,
  editVersion,
  selectedObjectId,
  editMode,
  editParams,
  isProcessing,
  zoom,
  currentFrame,
  totalFrames,
  frameWidth,
  frameHeight,
  onSelectObject,
  onUpload,
  onApplyEdit,
  onSegmentAtPoint,
}: EditorCanvasProps) {
  const [segmentMode, setSegmentMode] = useState(false);
  const imgRef = useRef<HTMLDivElement>(null);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (!segmentMode || !imgRef.current || !frameWidth || !frameHeight) {
        onSelectObject(null);
        return;
      }
      e.stopPropagation();
      const rect = imgRef.current.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width;
      const relY = (e.clientY - rect.top) / rect.height;
      const clickX = Math.round(relX * frameWidth);
      const clickY = Math.round(relY * frameHeight);
      onSegmentAtPoint(clickX, clickY);
      setSegmentMode(false);
    },
    [segmentMode, frameWidth, frameHeight, onSelectObject, onSegmentAtPoint]
  );

  if (!videoLoaded) {
    return <EmptyCanvas onUpload={onUpload} />;
  }

  const frameUrl = projectId
    ? `${API_URL}/frame/${projectId}/${currentFrame + 1}?v=${editVersion}`
    : null;

  return (
    <div
      className="flex-1 flex items-center justify-center overflow-hidden relative"
      style={{ background: "var(--ed-bg)" }}
    >
      {/* Segment toggle */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          setSegmentMode(!segmentMode);
        }}
        className="absolute top-4 right-4 z-30 flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium transition-all border"
        style={segmentMode ? {
          background: "var(--accent)",
          color: "#fff",
          borderColor: "var(--accent)",
          boxShadow: "0 4px 16px rgba(244,63,94,0.3)",
        } : {
          background: "var(--ed-surface)",
          color: "var(--ed-icon)",
          borderColor: "var(--ed-border)",
        }}
      >
        <Crosshair className="w-3.5 h-3.5" />
        {segmentMode ? "Click to segment" : "Segment"}
      </button>

      {isDetecting && (
        <div className="absolute inset-0 z-20 pointer-events-none animate-detection-shimmer" />
      )}

      <div
        className="relative"
        style={{ transform: `scale(${zoom / 100})`, transition: "transform 200ms ease" }}
      >
        <div
          ref={imgRef}
          className={`w-[768px] h-[432px] rounded-2xl overflow-hidden relative shadow-2xl ${segmentMode ? "cursor-crosshair" : ""}`}
          style={{
            background: "var(--ed-surface-2)",
            boxShadow: "0 25px 60px rgba(0,0,0,0.25)",
          }}
          onClick={handleCanvasClick}
        >
          {frameUrl ? (
            <img
              src={frameUrl}
              alt={`Frame ${currentFrame + 1}`}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-sm font-medium" style={{ color: "var(--ed-disabled)" }}>
                Frame Preview
              </div>
            </div>
          )}

          {segmentMode && (
            <div className="absolute inset-0 rounded-2xl border-2 border-[var(--accent)] pointer-events-none z-20" />
          )}

          {isProcessing && (
            <div className="absolute inset-0 flex items-center justify-center z-20 rounded-2xl" style={{ background: "rgba(0,0,0,0.5)" }}>
              <div className="flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                <span className="text-white text-sm font-medium">Applying edit…</span>
              </div>
            </div>
          )}

          {projectId && maskCount > 0 && !isSegmenting && (
            <img
              src={`${API_URL}/mask/${projectId}/${currentFrame + 1}?v=${maskVersion}`}
              alt="Segmentation mask"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none z-[1] mix-blend-screen opacity-40"
              style={{ filter: "hue-rotate(-30deg) saturate(3)" }}
            />
          )}

          {isSegmenting && (
            <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
              <div
                className="flex flex-col items-center gap-2 px-5 py-3.5 rounded-2xl border"
                style={{ background: "rgba(0,0,0,0.7)", borderColor: "rgba(255,255,255,0.1)" }}
              >
                <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                <span className="text-white/70 text-xs font-medium">Segmenting…</span>
              </div>
            </div>
          )}

          {!segmentMode && detections.map((det) => (
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
