"use client";

import { useCallback, useRef, useEffect } from "react";
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
  aiEditStatus: "idle" | "preview" | "applying" | "done";
  onSelectObject: (id: string | null) => void;
  onUpload: () => void;
  onApplyEdit: (action: EditAction, params: { color?: string; prompt?: string; scale?: number }) => void;
  onSegmentAtPoint: (clickX: number, clickY: number) => void;
  onCancelEdit: () => void;
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
  aiEditStatus,
  onSelectObject,
  onUpload,
  onApplyEdit,
  onSegmentAtPoint,
  onCancelEdit,
}: EditorCanvasProps) {
  const imgRef = useRef<HTMLDivElement>(null);
  const borderCanvasRef = useRef<HTMLCanvasElement>(null);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (!imgRef.current || !frameWidth || !frameHeight) {
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
    [frameWidth, frameHeight, onSegmentAtPoint]
  );

  // Draw border outline on canvas when mask exists
  useEffect(() => {
    if (!borderCanvasRef.current || !projectId || maskCount === 0 || isSegmenting || aiEditStatus === "preview") {
      return;
    }

    const canvas = borderCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size to match container
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Load mask image
    const maskImg = new Image();
    maskImg.crossOrigin = "anonymous";
    maskImg.onload = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Calculate scaling to fit mask in canvas
      const scaleX = canvas.width / maskImg.width;
      const scaleY = canvas.height / maskImg.height;
      const scale = Math.min(scaleX, scaleY);
      const x = (canvas.width - maskImg.width * scale) / 2;
      const y = (canvas.height - maskImg.height * scale) / 2;

      // Draw mask to get pixel data
      ctx.drawImage(maskImg, x, y, maskImg.width * scale, maskImg.height * scale);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Clear canvas again
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Find edges and draw glowing border
      ctx.strokeStyle = "rgba(244,63,94,1)";
      ctx.lineWidth = 3;
      ctx.shadowBlur = 8;
      ctx.shadowColor = "rgba(244,63,94,1)";

      // Simple edge detection - draw border where mask pixels meet non-mask pixels
      for (let y = 1; y < canvas.height - 1; y++) {
        for (let x = 1; x < canvas.width - 1; x++) {
          const idx = (y * canvas.width + x) * 4;
          const isMask = data[idx] > 128; // White pixel in mask

          // Check neighbors
          const topIdx = ((y - 1) * canvas.width + x) * 4;
          const bottomIdx = ((y + 1) * canvas.width + x) * 4;
          const leftIdx = (y * canvas.width + (x - 1)) * 4;
          const rightIdx = (y * canvas.width + (x + 1)) * 4;

          const topIsMask = data[topIdx] > 128;
          const bottomIsMask = data[bottomIdx] > 128;
          const leftIsMask = data[leftIdx] > 128;
          const rightIsMask = data[rightIdx] > 128;

          // Draw pixel if it's on the edge
          if (isMask && (!topIsMask || !bottomIsMask || !leftIsMask || !rightIsMask)) {
            ctx.fillStyle = "rgba(244,63,94,1)";
            ctx.fillRect(x, y, 1, 1);
          }
        }
      }
    };
    maskImg.src = `${API_URL}/mask/${projectId}/${currentFrame + 1}?v=${maskVersion}`;
  }, [projectId, currentFrame, maskCount, maskVersion, isSegmenting, aiEditStatus, frameWidth, frameHeight]);

  if (!videoLoaded) {
    return <EmptyCanvas onUpload={onUpload} />;
  }

  // Show preview frame if in preview mode, otherwise show current frame
  // Use per-frame versioning for transformed frames, otherwise use global editVersion
  const currentFrameIndex = currentFrame + 1; // Backend uses 1-based indexing
  const frameVersion = transformedFrameVersions?.[currentFrameIndex] ?? editVersion;
  const frameUrl = aiEditStatus === "preview" && previewFrameUrl
    ? previewFrameUrl
    : projectId
    ? `${API_URL}/frame/${projectId}/${currentFrameIndex}?v=${frameVersion}`
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
          {aiEditStatus !== "preview" && projectId && maskCount > 0 && !isSegmenting && (
            <>
              {/* Draw border outline on canvas - no mask overlay */}
              <canvas
                ref={borderCanvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none z-[2]"
                style={{ objectFit: "contain" }}
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
                <span className="text-white/70 text-xs font-medium">Segmenting…</span>
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
