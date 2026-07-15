"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type Detection,
  type EditMode,
  type EditParams,
  type FrameData,
  generateFrames,
} from "@/lib/mock-data";
import { useVideoStore } from "@/stores/videoStore";
import { useChangeLogStore } from "@/stores/changeLogStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SEGMENTING_STATUSES = new Set(["segmenting", "propagating"]);

interface PerFrameDetections {
  [frameKey: string]: { label: string; confidence: number; bbox: [number, number, number, number] }[];
}

interface EditorState {
  projectId: string | null;
  videoLoaded: boolean;
  videoName: string;
  duration: number;
  fps: number;
  frames: FrameData[];
  frameWidth: number;
  frameHeight: number;
  currentFrame: number;
  isPlaying: boolean;
  allDetections: PerFrameDetections;
  detections: Detection[];
  isDetecting: boolean;
  isSegmenting: boolean;
  maskCount: number;
  maskVersion: number;
  editVersion: number;
  transformedFrameVersions?: { [frameIndex: number]: number }; // Per-frame versioning for changed frames
  selectedObjectId: string | null;
  editMode: EditMode | null;
  editParams: EditParams;
  isProcessing: boolean;
  editProgress: { done: number; total: number };
  editStatus: "uploading" | "editing" | "done" | "error" | null;
  applyToAllFrames: boolean;
  editRangeStart: number;
  editRangeEnd: number;
  zoom: number;
  showEditPanel: boolean;
  toastMessage: string;
  showToast: boolean;
  aiPreviewFrameUrl: string | null;
  aiGenerationId: string | null;
  isAIGenerating: boolean;
  aiEditStatus: "idle" | "preview" | "applying" | "done";
  storageBaseUrl: string | null;
  aiEditProgress: { done: number; total: number };
  aiEditPhase: "transforming" | "interpolating" | "done" | null;
  aiInterpolationProgress: { done: number; total: number };
  isRefining: boolean;
  changeMarkers: Array<{
    id: string;
    frame: number;
    editType: string;
    timestamp: number;
    params?: { color?: string; prompt?: string; scale?: number; dx?: number; dy?: number };
  }>;
  isExporting: boolean;
}

const DEFAULT_EDIT_PARAMS: EditParams = {
  recolor: { color: "#F43F5E", opacity: 0.6 },
  resize: { scale: 1.0 },
  replace: { imageUrl: null },
};

export function useEditorState(projectId?: string, initialFrame = 0) {
  const updateProject = useVideoStore((state) => state.updateProject);
  const getProject = useVideoStore((state) => state.getProject);
  const [state, setState] = useState<EditorState>({
    projectId: projectId ?? null,
    videoLoaded: false,
    videoName: "",
    duration: 0,
    fps: 30,
    frames: [],
    frameWidth: 0,
    frameHeight: 0,
    currentFrame: initialFrame,
    isPlaying: false,
    allDetections: {},
    detections: [],
    isDetecting: false,
    isSegmenting: false,
    maskCount: 0,
    maskVersion: 0,
    editVersion: 0,
    selectedObjectId: null,
    editMode: null,
    editParams: DEFAULT_EDIT_PARAMS,
    isProcessing: false,
    editProgress: { done: 0, total: 0 },
    editStatus: null,
    applyToAllFrames: true,
    editRangeStart: 0,
    editRangeEnd: 0,
    zoom: 100,
    showEditPanel: false,
    toastMessage: "",
    showToast: false,
    aiPreviewFrameUrl: null,
    aiGenerationId: null,
    isAIGenerating: false,
    aiEditStatus: "idle",
    storageBaseUrl: null,
    aiEditProgress: { done: 0, total: 0 },
    aiEditPhase: null,
    aiInterpolationProgress: { done: 0, total: 0 },
    isRefining: false,
    changeMarkers: [],
    isExporting: false,
  });

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const acceptInProgressRef = useRef<boolean>(false);

  // Unified polling for project status - single interval, handles all status updates
  useEffect(() => {
    if (!projectId) return;

    const extractTriggered = { current: false };
    let lastAIProgressDone = 0;
    const startTime = Date.now();
    const MAX_POLL_TIME = 300000; // 5 minutes max polling time

    const poll = async () => {
      try {
        // Stop polling if we've been polling too long
        if (Date.now() - startTime > MAX_POLL_TIME) {
          console.warn("Polling timeout - stopping status checks");
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          setState((s) => ({
            ...s,
            showToast: true,
            toastMessage: "Video processing timed out. Please check your FFmpeg installation.",
          }));
          return;
        }

        const res = await fetch(`${API_URL}/project/${projectId}/status`);
        const status = await res.json();

        // Debug logging for segmentation status
        if (status.segment_status !== undefined || status.segmenting !== undefined) {
          console.log("[Frontend] Status poll - segmentation:", {
            segmenting: status.segmenting,
            segment_status: status.segment_status,
            mask_count: status.mask_count,
            fullStatus: status,
          });
        }

        // Update segmentation status immediately if present (before other status checks)
        // Also check mask_count to detect completed segmentation
        const hasMasksImmediate = (status.mask_count !== undefined && status.mask_count !== null && status.mask_count > 0);
        const segmentStatusImmediate = status.segment_status !== undefined
          ? status.segment_status
          : (hasMasksImmediate ? "done" : undefined);

        if (segmentStatusImmediate !== undefined) {
          const segmentingStatus = SEGMENTING_STATUSES.has(segmentStatusImmediate);
          const segmentError = segmentStatusImmediate === "error" ? status.segment_error : null;
          const isDone = segmentStatusImmediate === "done";

          console.log("[Frontend] Immediate segmentation status update:", {
            segment_status: segmentStatusImmediate,
            mask_count: status.mask_count,
            isDone,
            segmentingStatus,
            hasMasksImmediate,
          });

          setState((s) => {
            const newMaskCount = status.mask_count !== undefined && status.mask_count !== null
              ? status.mask_count
              : s.maskCount;
            // Increment maskVersion when done to force refresh
            const shouldIncrementMaskVersion = isDone && (newMaskCount > s.maskCount || s.maskVersion === 0);

            console.log("[Frontend] Updating segmentation state (immediate):", {
              oldMaskCount: s.maskCount,
              newMaskCount,
              oldMaskVersion: s.maskVersion,
              newMaskVersion: shouldIncrementMaskVersion ? s.maskVersion + 1 : s.maskVersion,
              isSegmenting: segmentingStatus,
              isDone,
            });

            return {
              ...s,
              isSegmenting: segmentingStatus,
              maskCount: newMaskCount,
              maskVersion: shouldIncrementMaskVersion ? s.maskVersion + 1 : s.maskVersion,
              showToast: segmentError ? true : s.showToast,
              toastMessage: segmentError ? `Segmentation failed: ${segmentError}` : s.toastMessage,
            };
          });
        }

        // Stop polling if there's an error
        if (status.status === "error") {
          console.error("Extraction error:", status.error);
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          setState((s) => ({
            ...s,
            showToast: true,
            toastMessage: `Extraction failed: ${status.error || "Unknown error"}`,
          }));
          return;
        }

        // Kick off extraction if project was just uploaded
        if ((status.status === "created" || status.status === "processing") && !extractTriggered.current) {
          extractTriggered.current = true;
          fetch(`${API_URL}/extract`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ project_id: projectId }),
          });
        }

        if (status.status === "ready" || status.status === "extracting") {
          const frameCount = status.frame_count || 0;
          if (frameCount > 0) {
            // Update Zustand store with video info
            const project = getProject(projectId);

            updateProject(projectId, {
              status: status.status,
              frameCount: frameCount,
              videoName: project?.videoName || projectId,
            });

            // Determine segmentation status - prioritize segment_status over segmenting boolean
            // Also check mask_count to restore segmentation state on page reload
            const maskCountFromStatus = status.mask_count !== undefined && status.mask_count !== null ? status.mask_count : 0;
            const hasMasks = maskCountFromStatus > 0;
            const isSegmenting = status.segment_status === "segmenting" || (status.segment_status === undefined && !!status.segmenting);

            // If masks exist but no segment_status, assume segmentation was done previously
            const segmentStatus = status.segment_status !== undefined
              ? status.segment_status
              : (hasMasks ? "done" : undefined);

            setState((s) => ({
              ...s,
              projectId,
              videoLoaded: true,
              videoName: project?.videoName || projectId,
              fps: 30,
              duration: frameCount / 30,
              frames: generateFrames(frameCount),
              frameWidth: status.frame_width || 0,
              frameHeight: status.frame_height || 0,
              editRangeEnd: s.editRangeEnd === 0 ? frameCount - 1 : s.editRangeEnd,
              isDetecting: !!status.detecting,
              // Use segmentStatus to determine isSegmenting - only true if actively segmenting
              isSegmenting: segmentStatus === "segmenting",
              maskCount: maskCountFromStatus,
              storageBaseUrl: status.storage_base_url || s.storageBaseUrl,
            }));

            // Store all per-frame detections
            if (status.detections && Object.keys(status.detections).length > 0) {
              setState((s) => ({ ...s, allDetections: status.detections }));
            }

            // Update segmentation status and handle errors (this runs after the initial state update)
            // Use the segmentStatus already calculated above
            if (segmentStatus !== undefined) {
            const segmentingStatus = SEGMENTING_STATUSES.has(segmentStatus);
              const segmentError = segmentStatus === "error" ? status.segment_error : null;
              const isDone = segmentStatus === "done";

              console.log("[Frontend] Segmentation status update:", {
                segment_status: segmentStatus,
                mask_count: status.mask_count,
                isDone,
                segmentingStatus,
                hasMasks,
              });

              setState((s) => {
                const newMaskCount = status.mask_count !== undefined && status.mask_count !== null
                  ? status.mask_count
                  : s.maskCount;
                // Increment maskVersion when segmentation completes to force mask refresh
                // Also increment on page reload if masks exist to ensure display refreshes
                const shouldIncrementMaskVersion = isDone && (newMaskCount > s.maskCount || s.maskVersion === 0);

                console.log("[Frontend] Updating segmentation state:", {
                  oldMaskCount: s.maskCount,
                  newMaskCount,
                  oldMaskVersion: s.maskVersion,
                  newMaskVersion: shouldIncrementMaskVersion ? s.maskVersion + 1 : s.maskVersion,
                  isSegmenting: segmentingStatus,
                });

                return {
                  ...s,
                  isSegmenting: segmentingStatus,
                  maskCount: newMaskCount,
                  maskVersion: shouldIncrementMaskVersion ? s.maskVersion + 1 : s.maskVersion,
                  showToast: segmentError ? true : s.showToast,
                  toastMessage: segmentError ? `Segmentation failed: ${segmentError}` : s.toastMessage,
                };
              });
            }

            // Update local frame-edit status (recolor, remove, replace, etc.)
            if (status.edit_status !== undefined) {
              const editDone = status.edit_status === "done";
              const editError = status.edit_status === "error";
              const editCancelled = status.edit_status === "cancelled";
              const editProcessing = ["uploading", "editing", "processing"].includes(status.edit_status);
              const editProgress = status.edit_progress || { done: 0, total: 0 };
              const backendEditVersion = Number(status.edit_version || 0);

              setState((s) => {
                // Only update if still processing — prevents repeated editVersion increments
                if (!s.isProcessing && (editDone || editError || editCancelled) && backendEditVersion <= s.editVersion) return s;
                return {
                  ...s,
                  isProcessing: editProcessing,
                  editProgress: editProgress,
                  editStatus: editProcessing
                    ? status.edit_status === "processing" ? "editing" : status.edit_status as "uploading" | "editing"
                    : editDone ? "done" : editError ? "error" : null,
                  editVersion: Math.max(
                    s.editVersion,
                    backendEditVersion,
                    editDone && s.isProcessing ? s.editVersion + 1 : s.editVersion,
                  ),
                  showToast: (editDone || editError) && s.isProcessing ? true : s.showToast,
                  toastMessage: editDone && s.isProcessing
                    ? "Edit applied successfully"
                    : editError && s.isProcessing
                      ? `Edit failed: ${status.edit_error || "Unknown error"}`
                      : s.toastMessage,
                };
              });
            }

            // Update refine status (Gemini photorealistic refinement)
            if (status.refine_status !== undefined) {
              const refineProcessing = status.refine_status === "processing";
              const refineDone = status.refine_status === "done";
              const refineError = status.refine_status === "error";
              setState((s) => {
                return {
                  ...s,
                  isRefining: refineProcessing,
                  isProcessing: refineProcessing || (s.isProcessing && !refineDone && !refineError),
                  editVersion: refineDone && s.isRefining ? s.editVersion + 1 : s.editVersion,
                  showToast: (refineDone || refineError) && s.isRefining ? true : s.showToast,
                  toastMessage: refineDone && s.isRefining ? "Frame refined successfully"
                    : refineError && s.isRefining ? `Refinement failed: ${status.refine_error || "Unknown error"}`
                      : s.toastMessage,
                };
              });
            }

            // Update AI edit status and progress
            if (status.ai_edit_status !== undefined) {
              const aiProgress = status.ai_edit_progress || { done: 0, total: 0 };
              const aiInterpolationProgress = status.ai_interpolation_progress || { done: 0, total: 0 };
              const aiEditPhase = status.ai_edit_phase || null;
              const isDone = status.ai_edit_status === "done" || status.ai_edit_status === "error";
              const transformedFrames = status.ai_edit_transformed_frames || [];

              // Track transformed frames for per-frame versioning
              // This allows us to only refresh frames that were actually changed
              if (isDone && transformedFrames.length > 0) {
                lastAIProgressDone = aiProgress.total; // Mark as processed
                setState((s) => ({
                  ...s,
                  transformedFrameVersions: {
                    ...(s.transformedFrameVersions || {}),
                    ...Object.fromEntries(transformedFrames.map((f: number) => [f, (s.transformedFrameVersions?.[f] || 0) + 1]))
                  }
                }));
              }

              setState((s) => ({
                ...s,
                aiEditStatus: status.ai_edit_status === "processing" ? "applying" :
                  status.ai_edit_status === "done" ? "done" :
                    status.ai_edit_status === "preview" ? "preview" :
                      status.ai_edit_status === "error" ? "idle" : s.aiEditStatus,
                aiEditProgress: aiProgress,
                aiEditPhase: aiEditPhase,
                aiInterpolationProgress: aiInterpolationProgress,
                aiPreviewFrameUrl: isDone ? null : (status.ai_preview_url ? `${API_URL}${status.ai_preview_url}` : s.aiPreviewFrameUrl),
                aiGenerationId: isDone ? null : (status.ai_generation_id || s.aiGenerationId),
                showToast: isDone ? true : s.showToast,
                toastMessage: isDone ? (status.ai_edit_status === "done"
                  ? "AI edit applied successfully"
                  : `AI edit failed: ${status.ai_edit_error || "unknown error"}`) : s.toastMessage,
              }));

              if (isDone) {
                acceptInProgressRef.current = false;
              }
            }

            // Continue polling if any operation is in progress
            const shouldStopPolling = status.status === "ready" &&
              !status.detecting &&
              !status.segmenting &&
              status.edit_status !== "uploading" &&
              status.edit_status !== "editing" &&
              status.edit_status !== "processing" &&
              status.segment_status !== "propagating" &&
              status.refine_status !== "processing" &&
              status.ai_edit_status !== "processing" &&
              status.ai_edit_status !== "applying";

            if (shouldStopPolling) {
              if (pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
              }
            } else if (!pollingRef.current) {
              // Restart polling if it was stopped but we need it again
              pollingRef.current = setInterval(poll, 2000);
            }
          }
        }
      } catch {
        // Backend not reachable yet, keep polling
      }
    };

    // Clear any existing polling first
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    poll();
    // Use longer interval - 2000ms instead of 1500ms, and only poll when needed
    pollingRef.current = setInterval(poll, 2000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [projectId]);

  // Update displayed detections when frame changes
  useEffect(() => {
    const frameKey = String(state.currentFrame + 1); // backend uses 1-based keys
    const frameDets = state.allDetections[frameKey] || [];
    const mapped: Detection[] = frameDets.map((d, i) => ({
      id: `obj-${i}`,
      label: d.label,
      confidence: d.confidence,
      bbox: d.bbox,
    }));
    setState((s) => ({ ...s, detections: mapped, selectedObjectId: null, showEditPanel: false }));
  }, [state.currentFrame, state.allDetections]);

  const loadVideo = useCallback(() => {
    // No-op when using real backend — video loads via polling
  }, []);

  const detectObjects = useCallback(() => {
    if (!state.projectId) return;
    setState((s) => ({ ...s, isDetecting: true }));
    // Detections are loaded via polling from the backend
    // They run automatically during /extract
  }, [state.projectId]);

  const restartPolling = useCallback(() => {
    if (!pollingRef.current && projectId) {
      const poll = async () => {
        try {
          const res = await fetch(`${API_URL}/project/${projectId}/status`);
          const status = await res.json();

          // Handle segmentation status
          const hasMasks = (status.mask_count !== undefined && status.mask_count !== null && status.mask_count > 0);
          const segStatus = status.segment_status !== undefined
            ? status.segment_status
            : (hasMasks ? "done" : undefined);

          if (segStatus !== undefined) {
            const isDone = segStatus === "done";
            setState((s) => {
              const newMaskCount = status.mask_count !== undefined && status.mask_count !== null
                ? status.mask_count : s.maskCount;
              const shouldIncrementMaskVersion = isDone && (newMaskCount > s.maskCount || s.maskVersion === 0);
              return {
                ...s,
                isSegmenting: SEGMENTING_STATUSES.has(segStatus),
                maskCount: newMaskCount,
                maskVersion: shouldIncrementMaskVersion ? s.maskVersion + 1 : s.maskVersion,
                showToast: status.segment_status === "error" ? true : s.showToast,
                toastMessage: status.segment_status === "error"
                  ? `Segmentation failed: ${status.segment_error}`
                  : s.toastMessage,
              };
            });
          }

          // Handle local frame-edit status
          if (status.edit_status !== undefined) {
            const editDone = status.edit_status === "done";
            const editError = status.edit_status === "error";
            const editCancelled = status.edit_status === "cancelled";
            const editProcessing = ["uploading", "editing", "processing"].includes(status.edit_status);
            const editProgress = status.edit_progress || { done: 0, total: 0 };
            const backendEditVersion = Number(status.edit_version || 0);
            setState((s) => {
              if (!s.isProcessing && (editDone || editError || editCancelled) && backendEditVersion <= s.editVersion) return s;
              return {
                ...s,
                isProcessing: editProcessing,
                editProgress,
                editStatus: editProcessing
                  ? status.edit_status === "processing" ? "editing" : status.edit_status as "uploading" | "editing"
                  : editDone ? "done" : editError ? "error" : null,
                editVersion: Math.max(
                  s.editVersion,
                  backendEditVersion,
                  editDone && s.isProcessing ? s.editVersion + 1 : s.editVersion,
                ),
                showToast: (editDone || editError) && s.isProcessing ? true : s.showToast,
                toastMessage: editDone && s.isProcessing ? "Edit applied successfully"
                  : editError && s.isProcessing ? `Edit failed: ${status.edit_error || "Unknown error"}`
                    : s.toastMessage,
              };
            });
          }

          // Handle refine status (Gemini photorealistic refinement)
          if (status.refine_status !== undefined) {
            const refineProcessing = status.refine_status === "processing";
            const refineDone = status.refine_status === "done";
            const refineError = status.refine_status === "error";
            setState((s) => {
              return {
                ...s,
                isRefining: refineProcessing,
                isProcessing: refineProcessing || (s.isProcessing && !refineDone && !refineError),
                editVersion: refineDone && s.isRefining ? s.editVersion + 1 : s.editVersion,
                showToast: (refineDone || refineError) && s.isRefining ? true : s.showToast,
                toastMessage: refineDone && s.isRefining ? "Frame refined successfully"
                  : refineError && s.isRefining ? `Refinement failed: ${status.refine_error || "Unknown error"}`
                    : s.toastMessage,
              };
            });
          }

          // Handle AI edit status (propagate / AI edit pipeline)
          if (status.ai_edit_status !== undefined) {
            const aiDone = status.ai_edit_status === "done" || status.ai_edit_status === "error";
            setState((s) => ({
              ...s,
              aiEditStatus: status.ai_edit_status === "processing" ? "applying" :
                status.ai_edit_status === "done" ? "done" :
                  status.ai_edit_status === "error" ? "idle" : s.aiEditStatus,
              aiEditProgress: status.ai_edit_progress || s.aiEditProgress,
              aiEditPhase: status.ai_edit_phase || s.aiEditPhase,
              aiInterpolationProgress: status.ai_interpolation_progress || s.aiInterpolationProgress,
              showToast: aiDone ? true : s.showToast,
              toastMessage: aiDone ? (status.ai_edit_status === "done"
                ? "Edit propagated to all frames"
                : `Propagation failed: ${status.ai_edit_error || "unknown"}`) : s.toastMessage,
            }));
          }

          // Stop polling when all operations are done
          const allDone = !status.segmenting &&
            status.segment_status !== "segmenting" &&
            status.segment_status !== "propagating" &&
            status.edit_status !== "uploading" &&
            status.edit_status !== "editing" &&
            status.edit_status !== "processing" &&
            status.refine_status !== "processing" &&
            status.ai_edit_status !== "processing";

          if (allDone && pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        } catch {
          // Backend not reachable, keep polling
        }
      };
      poll();
      pollingRef.current = setInterval(poll, 1500);
    }
  }, [projectId]);

  const segmentAtPoint = useCallback((clickX: number, clickY: number) => {
    const currentProjectId = projectId ?? state.projectId;
    if (!currentProjectId || state.aiEditStatus === "preview" || state.isProcessing || state.isSegmenting) return;

    const frameIndex = state.currentFrame + 1;
    const { addLog } = useChangeLogStore.getState();
    addLog(currentProjectId, {
      projectId: currentProjectId,
      type: "segment",
      frameIndex: state.currentFrame,
      data: { clickX, clickY },
    });

    // Keep network I/O outside setState. React may invoke state updaters more
    // than once in development, which previously submitted duplicate requests.
    fetch(`${API_URL}/segment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: currentProjectId,
        frame_index: frameIndex,
        click_x: clickX,
        click_y: clickY,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          console.error("Segmentation error:", data.error);
          setState((prev) => ({
            ...prev,
            isSegmenting: false,
            showToast: true,
            toastMessage: `Segmentation failed: ${data.error}`,
          }));
        }
        restartPolling();
      })
      .catch((err) => {
        console.error("Segmentation error:", err);
        setState((prev) => ({
          ...prev,
          isSegmenting: false,
          showToast: true,
          toastMessage: `Segmentation failed: ${err.message}`,
        }));
      });

    setState((s) => ({
      ...s,
      isSegmenting: true,
      isProcessing: false,
      maskCount: 0,
      selectedObjectId: null,
      showEditPanel: false,
    }));
  }, [projectId, restartPolling, state.aiEditStatus, state.currentFrame, state.isProcessing, state.isSegmenting, state.projectId]);

  const selectObject = useCallback((id: string | null) => {
    if (id !== null && state.frameWidth > 0 && state.frameHeight > 0) {
      const detection = state.detections.find((d) => d.id === id);
      if (detection) {
        const [xPct, yPct, wPct, hPct] = detection.bbox;
        const clickX = Math.round(((xPct + wPct / 2) / 100) * state.frameWidth);
        const clickY = Math.round(((yPct + hPct / 2) / 100) * state.frameHeight);
        segmentAtPoint(clickX, clickY);
      }
    }

    setState((s) => ({
      ...s,
      selectedObjectId: id,
      showEditPanel: id !== null,
      editMode: id !== null ? "recolor" : null,
      isSegmenting: id !== null || s.isSegmenting,
      maskCount: id !== null ? 0 : s.maskCount,
    }));
  }, [segmentAtPoint, state.detections, state.frameHeight, state.frameWidth]);

  const setEditMode = useCallback((mode: EditMode) => {
    setState((s) => ({ ...s, editMode: mode }));
  }, []);

  const updateEditParams = useCallback(
    (mode: EditMode, params: Partial<EditParams[EditMode]>) => {
      setState((s) => ({
        ...s,
        editParams: {
          ...s.editParams,
          [mode]: { ...s.editParams[mode], ...params },
        },
      }));
    },
    []
  );

  const applyEdit = useCallback(() => {
    // Legacy — kept for interface compat
  }, []);

  const cancelEdit = useCallback(() => {
    setState((s) => {
      if (!s.projectId) return s;
      fetch(`${API_URL}/edit/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: s.projectId }),
      });
      return { ...s, isProcessing: false, aiEditStatus: "idle" as const, aiEditPhase: null };
    });
  }, []);

  const applyEditAction = useCallback(
    (action: string, params: { color?: string; prompt?: string; scale?: number; dx?: number; dy?: number }) => {
      const current = state;
      if (!current.projectId || current.isProcessing || current.isSegmenting) return;

      const MASK_ACTIONS = new Set(["delete", "replace", "resize", "blur_region", "recolor", "move", "color_pop", "glow"]);
      const isMaskEdit = MASK_ACTIONS.has(action);
      const startFrame = current.editRangeStart > 0
        ? current.editRangeStart + 1
        : (isMaskEdit ? 1 : current.currentFrame + 1);
      const endFrame = current.editRangeEnd > 0
        ? current.editRangeEnd + 1
        : (isMaskEdit ? current.frames.length : current.currentFrame + 1);
      const editRule: Record<string, unknown> = {
        edit_type: action,
        start_frame: startFrame,
        end_frame: endFrame,
      };
      if (params.color) editRule.color = params.color;
      if (params.prompt) editRule.prompt = params.prompt;
      if (params.scale) editRule.scale = params.scale;
      if (params.dx !== undefined) editRule.dx = params.dx;
      if (params.dy !== undefined) editRule.dy = params.dy;

      const { addLog } = useChangeLogStore.getState();
      addLog(current.projectId, {
        projectId: current.projectId,
        type: "edit",
        frameIndex: current.currentFrame,
        data: {
          editType: action,
          color: params.color,
          prompt: params.prompt,
          scale: params.scale,
          startFrame: startFrame - 1,
          endFrame: endFrame - 1,
        },
      });

      const markerId = `marker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newMarker = {
        id: markerId,
        frame: current.currentFrame,
        editType: action,
        timestamp: Date.now(),
        params: { ...params },
      };

      // Submit once, outside the state updater. The backend owns propagation;
      // Save/Export is only for producing the final MP4.
      fetch(`${API_URL}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: current.projectId, edit_rules: [editRule] }),
      })
        .then(async (res) => {
          const data = await res.json();
          if (!res.ok || data.error) {
            throw new Error(data.error || `Edit request failed (${res.status})`);
          }
          restartPolling();
        })
        .catch((err) => {
          setState((s) => ({
            ...s,
            isProcessing: false,
            editStatus: "error",
            showToast: true,
            toastMessage: `Edit failed: ${err.message}`,
          }));
        });

      setState((s) => ({
        ...s,
        isProcessing: true,
        editStatus: "editing",
        editProgress: { done: 0, total: endFrame - startFrame + 1 },
        selectedObjectId: null,
        showEditPanel: false,
        changeMarkers: [...s.changeMarkers, newMarker],
      }));
    },
    [restartPolling, state]
  );

  const handleMarkerDrag = useCallback((markerId: string, newFrame: number) => {
    const current = state;
    const marker = current.changeMarkers.find((m) => m.id === markerId);
    if (!marker || !current.projectId || current.isProcessing) return;

    const startFrame = newFrame;
    const endFrame = current.editRangeEnd > 0 ? current.editRangeEnd : current.frames.length - 1;
    const editRule: Record<string, unknown> = {
      edit_type: marker.editType,
      start_frame: startFrame + 1,
      end_frame: endFrame + 1,
      ...marker.params,
    };

    fetch(`${API_URL}/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: current.projectId, edit_rules: [editRule] }),
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `Edit request failed (${res.status})`);
        restartPolling();
      })
      .catch((err) => {
        setState((s) => ({
          ...s,
          isProcessing: false,
          editStatus: "error",
          showToast: true,
          toastMessage: `Edit failed: ${err.message}`,
        }));
      });

    setState((s) => ({
      ...s,
      changeMarkers: s.changeMarkers.map((m) =>
        m.id === markerId ? { ...m, frame: newFrame } : m
      ),
      editRangeStart: startFrame,
      editRangeEnd: endFrame,
      isProcessing: true,
      editStatus: "editing",
    }));
  }, [restartPolling, state]);

  const undoEdit = useCallback(() => {
    setState((s) => {
      if (!s.projectId) return s;

      fetch(`${API_URL}/edit/undo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.error) {
            setState((s) => ({
              ...s,
              showToast: true,
              toastMessage: `Undo failed: ${data.error}`,
            }));
          } else {
            restartPolling();
            setState((s) => ({
              ...s,
              editVersion: s.editVersion + 1,
              showToast: true,
              toastMessage: data.message || "Edit undone successfully",
            }));
          }
        })
        .catch((err) => {
          setState((s) => ({
            ...s,
            showToast: true,
            toastMessage: `Undo failed: ${err.message}`,
          }));
        });

      return { ...s };
    });
  }, [restartPolling]);

  const setCurrentFrame = useCallback((frame: number) => {
    setState((s) => {
      // Clear preview when changing frames (unless we're applying the edit)
      const shouldClearPreview = s.aiEditStatus === "preview" && s.currentFrame !== frame;
      return {
        ...s,
        currentFrame: frame,
        // Clear preview when navigating away
        aiPreviewFrameUrl: shouldClearPreview ? null : s.aiPreviewFrameUrl,
        aiEditStatus: shouldClearPreview ? "idle" : s.aiEditStatus,
      };
    });
  }, []);

  const togglePlay = useCallback(() => {
    setState((s) => ({ ...s, isPlaying: !s.isPlaying }));
  }, []);

  const setZoom = useCallback((zoom: number) => {
    setState((s) => ({ ...s, zoom }));
  }, []);

  const setApplyToAllFrames = useCallback((value: boolean) => {
    setState((s) => ({ ...s, applyToAllFrames: value }));
  }, []);

  const setEditRange = useCallback((start: number, end: number) => {
    setState((s) => ({ ...s, editRangeStart: start, editRangeEnd: end }));
  }, []);

  const closeEditPanel = useCallback(() => {
    setState((s) => ({
      ...s,
      selectedObjectId: null,
      showEditPanel: false,
      editMode: null,
    }));
  }, []);

  const setVideoName = useCallback((name: string) => {
    setState((s) => ({ ...s, videoName: name }));
  }, []);

  const hideToast = useCallback(() => {
    setState((s) => ({ ...s, showToast: false }));
  }, []);

  const selectedObject = useMemo(
    () => state.detections.find((d) => d.id === state.selectedObjectId) ?? null,
    [state.detections, state.selectedObjectId]
  );

  const exportToMp4 = useCallback(async () => {
    const pid = projectId ?? state.projectId;
    if (!pid || !state.videoLoaded) {
      setState((s) => ({ ...s, showToast: true, toastMessage: "No project or video loaded" }));
      return;
    }
    setState((s) => ({ ...s, isExporting: true }));
    try {
      const res = await fetch(`${API_URL}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: pid }),
      });
      const data = await res.json();
      if (data.error) {
        setState((s) => ({ ...s, isExporting: false, showToast: true, toastMessage: data.error }));
        return;
      }
      const videoRes = await fetch(`${API_URL}/render/${pid}/video`);
      if (!videoRes.ok) {
        setState((s) => ({ ...s, isExporting: false, showToast: true, toastMessage: "Failed to download video" }));
        return;
      }
      const blob = await videoRes.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(state.videoName || "export").replace(/\.[^.]+$/, "")}.mp4`;
      a.click();
      URL.revokeObjectURL(url);
      setState((s) => ({ ...s, isExporting: false, showToast: true, toastMessage: "Video exported as MP4" }));
    } catch (err) {
      setState((s) => ({
        ...s,
        isExporting: false,
        showToast: true,
        toastMessage: err instanceof Error ? err.message : "Export failed",
      }));
    }
  }, [projectId, state.projectId, state.videoLoaded, state.videoName]);

  return {
    ...state,
    selectedObject,
    loadVideo,
    detectObjects,
    segmentAtPoint,
    selectObject,
    setEditMode,
    updateEditParams,
    applyEdit,
    cancelEdit,
    applyEditAction,
    undoEdit,
    setCurrentFrame,
    togglePlay,
    setZoom,
    setApplyToAllFrames,
    setEditRange,
    closeEditPanel,
    setVideoName,
    hideToast,
    editProgress: state.editProgress,
    editStatus: state.editStatus,
    changeMarkers: state.changeMarkers,
    handleMarkerDrag,
    exportToMp4,
    isExporting: state.isExporting,
  };
}
