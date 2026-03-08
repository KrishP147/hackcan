"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type Detection,
  type EditMode,
  type EditParams,
  type FrameData,
  generateFrames,
} from "@/lib/mock-data";
import type { ChatMessage } from "@/components/editor/AIChatPane";
import { useVideoStore } from "@/stores/videoStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  aiChatHistory: ChatMessage[];
  aiPreviewFrameUrl: string | null;
  aiGenerationId: string | null;
  isAIGenerating: boolean;
  aiEditStatus: "idle" | "preview" | "applying" | "done";
  aiEditProgress: { done: number; total: number };
  aiEditPhase: "transforming" | "interpolating" | "done" | null;
  aiInterpolationProgress: { done: number; total: number };
  isRefining: boolean;
  changeMarkers: Array<{ id: string; frame: number; editType: string; timestamp: number }>;
}

const DEFAULT_EDIT_PARAMS: EditParams = {
  recolor: { color: "#F43F5E", opacity: 0.6 },
  resize: { scale: 1.0 },
  replace: { imageUrl: null },
};

export function useEditorState(projectId?: string) {
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
    currentFrame: 0,
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
    aiChatHistory: [],
    aiPreviewFrameUrl: null,
    aiGenerationId: null,
    isAIGenerating: false,
    aiEditStatus: "idle",
    aiEditProgress: { done: 0, total: 0 },
    aiEditPhase: null,
    aiInterpolationProgress: { done: 0, total: 0 },
    isRefining: false,
    changeMarkers: [],
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
          const segmentingStatus = segmentStatusImmediate === "segmenting";
          const segmentError = status.segment_error;
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
            }));

            // Store all per-frame detections
            if (status.detections && Object.keys(status.detections).length > 0) {
              setState((s) => ({ ...s, allDetections: status.detections }));
            }

            // Update segmentation status and handle errors (this runs after the initial state update)
            // Use the segmentStatus already calculated above
            if (segmentStatus !== undefined) {
              const segmentingStatus = segmentStatus === "segmenting";
              const segmentError = status.segment_error;
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

            // Update edit status (Cloudinary edits like recolor, remove, replace, etc.)
            if (status.edit_status !== undefined) {
              const editDone = status.edit_status === "done";
              const editError = status.edit_status === "error";
              const editCancelled = status.edit_status === "cancelled";
              const editProgress = status.edit_progress || { done: 0, total: 0 };

              setState((s) => {
                // Only update if still processing — prevents repeated editVersion increments
                if (!s.isProcessing && (editDone || editError || editCancelled)) return s;
                return {
                  ...s,
                  isProcessing: !(editDone || editError || editCancelled),
                  editProgress: editProgress,
                  editStatus: status.edit_status as "uploading" | "editing" | "done" | "error" | null,
                  editVersion: (editDone && s.isProcessing) ? s.editVersion + 1 : s.editVersion,
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
                isSegmenting: segStatus === "segmenting",
                maskCount: newMaskCount,
                maskVersion: shouldIncrementMaskVersion ? s.maskVersion + 1 : s.maskVersion,
                showToast: status.segment_error ? true : s.showToast,
                toastMessage: status.segment_error ? `Segmentation failed: ${status.segment_error}` : s.toastMessage,
              };
            });
          }

          // Handle edit status (Cloudinary edits)
          if (status.edit_status !== undefined) {
            const editDone = status.edit_status === "done";
            const editError = status.edit_status === "error";
            const editCancelled = status.edit_status === "cancelled";
            setState((s) => {
              if (!s.isProcessing && (editDone || editError || editCancelled)) return s;
              return {
                ...s,
                isProcessing: !(editDone || editError || editCancelled),
                editVersion: (editDone && s.isProcessing) ? s.editVersion + 1 : s.editVersion,
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
            status.edit_status !== "uploading" &&
            status.edit_status !== "editing" &&
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
    setState((s) => {
      if (!s.projectId) return s;

      fetch(`${API_URL}/segment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          frame_index: s.currentFrame + 1,
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
          // Restart polling to pick up segmentation completion
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

      return {
        ...s,
        isSegmenting: true,
        isProcessing: false,
        maskCount: 0,
        selectedObjectId: null,
        showEditPanel: false,
      };
    });
  }, [restartPolling]);

  const selectObject = useCallback((id: string | null) => {
    setState((s) => {
      // Trigger segmentation when selecting an object
      if (id !== null && s.projectId && s.frameWidth > 0) {
        const det = s.detections.find((d) => d.id === id);
        if (det) {
          const [xPct, yPct, wPct, hPct] = det.bbox;
          const clickX = Math.round(((xPct + wPct / 2) / 100) * s.frameWidth);
          const clickY = Math.round(((yPct + hPct / 2) / 100) * s.frameHeight);

          // Segmentation disabled - no-op
        }
      }

      return {
        ...s,
        selectedObjectId: id,
        showEditPanel: id !== null,
        editMode: id !== null ? "recolor" : null,
        isSegmenting: id !== null,
        maskCount: 0,
      };
    });
  }, []);

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
    (action: string, params: { color?: string; prompt?: string; scale?: number }) => {
      setState((s) => {
        if (!s.projectId) return s;

        // Use edit range if explicitly set, otherwise just the current frame
        const startFrame = s.editRangeStart > 0 ? s.editRangeStart + 1 : s.currentFrame + 1;
        const endFrame = s.editRangeEnd > 0 ? s.editRangeEnd + 1 : s.currentFrame + 1;
        const editRule: Record<string, unknown> = {
          edit_type: action,
          start_frame: startFrame,
          end_frame: endFrame,
        };
        if (params.color) editRule.color = params.color;
        if (params.prompt) editRule.prompt = params.prompt;
        if (params.scale) editRule.scale = params.scale;

        // Add change marker at current frame
        const markerId = `marker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newMarker = {
          id: markerId,
          frame: s.currentFrame,
          editType: action,
          timestamp: Date.now(),
        };

        fetch(`${API_URL}/edit`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: s.projectId,
            edit_rules: [editRule],
          }),
        }).then(() => {
          // Restart polling to track edit progress
          restartPolling();
        });

        return { 
          ...s, 
          isProcessing: true, 
          selectedObjectId: null, 
          showEditPanel: false,
          changeMarkers: [...s.changeMarkers, newMarker],
        };
      });
    },
    [restartPolling]
  );

  const refineFrame = useCallback(() => {
    setState((s) => {
      if (!s.projectId) return s;

      // Add change marker for refine
      const markerId = `marker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newMarker = {
        id: markerId,
        frame: s.currentFrame,
        editType: "refine",
        timestamp: Date.now(),
      };

      fetch(`${API_URL}/edit/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          frame_index: s.currentFrame + 1,
        }),
      }).then(() => {
        restartPolling();
      });

      return { 
        ...s, 
        isProcessing: true,
        changeMarkers: [...s.changeMarkers, newMarker],
      };
    });
  }, [restartPolling]);

  const propagateEdit = useCallback((prompt: string) => {
    setState((s) => {
      if (!s.projectId) return s;

      fetch(`${API_URL}/edit/propagate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          frame_index: s.currentFrame + 1,
          prompt,
          start_frame: s.editRangeStart > 0 ? s.editRangeStart + 1 : 1,
          end_frame: s.editRangeEnd > 0 ? s.editRangeEnd + 1 : 0,
        }),
      }).then(() => {
        restartPolling();
      });

      return {
        ...s,
        aiEditStatus: "applying",
        aiEditPhase: "transforming",
        aiEditProgress: { done: 0, total: 0 },
        aiInterpolationProgress: { done: 0, total: 0 },
      };
    });
  }, [restartPolling]);

  const handleMarkerDrag = useCallback((markerId: string, newFrame: number) => {
    setState((s) => {
      // Find the marker
      const marker = s.changeMarkers.find((m) => m.id === markerId);
      if (!marker || !s.projectId) return s;

      // Update marker position
      const updatedMarkers = s.changeMarkers.map((m) =>
        m.id === markerId ? { ...m, frame: newFrame } : m
      );

      // Update edit range to propagate from marker frame to end
      const startFrame = newFrame;
      const endFrame = s.editRangeEnd > 0 ? s.editRangeEnd : s.frames.length - 1;
      
      // Trigger propagation based on edit type
      if (marker.editType === "refine") {
        // For refine, propagate the realistic enhancement
        fetch(`${API_URL}/edit/propagate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: s.projectId,
            frame_index: newFrame + 1,
            prompt: "Apply the same realistic enhancement consistently",
            start_frame: startFrame + 1,
            end_frame: endFrame + 1,
            interval: 60,
          }),
        }).then(() => {
          restartPolling();
        });
      } else {
        // For other edits, propagate the edit type
        const editRule: Record<string, unknown> = {
          edit_type: marker.editType,
          start_frame: startFrame + 1,
          end_frame: endFrame + 1,
        };
        
        fetch(`${API_URL}/edit`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: s.projectId,
            edit_rules: [editRule],
          }),
        }).then(() => {
          restartPolling();
        });
      }

      return {
        ...s,
        changeMarkers: updatedMarkers,
        editRangeStart: startFrame,
        editRangeEnd: endFrame,
        isProcessing: true,
      };
    });
  }, [restartPolling]);

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

  const sendAIPrompt = useCallback((prompt: string) => {
    setState((s) => {
      if (!s.projectId) return s;

      // Add user message to chat history
      const userMessage: ChatMessage = {
        role: "user",
        message: prompt,
        timestamp: Date.now(),
      };

      setState((prev) => ({
        ...prev,
        aiChatHistory: [...prev.aiChatHistory, userMessage],
        isAIGenerating: true,
        aiEditStatus: "idle",
      }));

      // Call preview endpoint
      fetch(`${API_URL}/ai/edit/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          frame_index: s.currentFrame + 1,
          prompt: prompt,
        }),
      })
        .then(async (res) => {
          if (!res.ok) {
            const errorData = await res.json().catch(() => ({ error: "Unknown error" }));
            throw new Error(errorData.error || `HTTP ${res.status}`);
          }
          return res.json();
        })
        .then((data) => {
          if (data.error) {
            throw new Error(data.error);
          }
          const previewUrl = `${API_URL}${data.preview_url}`;
          console.log("Preview URL:", previewUrl);
          const assistantMessage: ChatMessage = {
            role: "assistant",
            message: "Preview generated! Review it in the canvas.",
            timestamp: Date.now(),
          };
          setState((prev) => ({
            ...prev,
            aiChatHistory: [...prev.aiChatHistory, assistantMessage],
            aiPreviewFrameUrl: previewUrl,
            aiGenerationId: data.generation_id,
            isAIGenerating: false,
            aiEditStatus: "preview",
          }));
        })
        .catch((err) => {
          console.error("AI preview error:", err);
          setState((prev) => ({
            ...prev,
            isAIGenerating: false,
            aiEditStatus: "idle",
            showToast: true,
            toastMessage: `Failed to generate preview: ${err.message}`,
          }));
        });

      return s;
    });
  }, []);

  const acceptAIGeneration = useCallback(() => {
    // Use ref to prevent duplicate calls (faster than state check)
    if (acceptInProgressRef.current) {
      console.log("Accept already in progress (ref check), ignoring duplicate call");
      return;
    }

    setState((s) => {
      if (!s.projectId || !s.aiGenerationId) {
        console.log("Cannot accept: missing projectId or generationId");
        return s;
      }
      // Prevent multiple calls - if already applying, ignore
      if (s.aiEditStatus === "applying") {
        console.log("Already applying, ignoring duplicate accept call");
        return s;
      }

      // Set ref immediately to prevent race conditions
      acceptInProgressRef.current = true;

      const startFrame = s.editRangeStart > 0 ? s.editRangeStart + 1 : s.currentFrame + 1;
      const endFrame = s.editRangeEnd > 0 ? s.editRangeEnd + 1 : s.frames.length;
      const interval = 60; // Transform every 60th frame from start to end

      const generationId = s.aiGenerationId; // Save before clearing

      // Set status immediately to prevent duplicate calls and clear generation ID
      console.log("[acceptAIGeneration] Setting status to 'applying'");
      setState((prev) => {
        const newState = {
          ...prev,
          aiEditStatus: "applying" as const,
          aiPreviewFrameUrl: null, // Clear preview when starting to apply
          aiGenerationId: null, // Clear generation ID to prevent duplicate calls
          aiEditProgress: { done: 0, total: 0 }, // Reset progress
          aiEditPhase: "transforming" as const,
          aiInterpolationProgress: { done: 0, total: 0 },
        };
        console.log("[acceptAIGeneration] New state:", { 
          aiEditStatus: newState.aiEditStatus, 
          aiEditProgress: newState.aiEditProgress 
        });
        return newState;
      });

      console.log("Calling accept with generation_id:", generationId);
      fetch(`${API_URL}/ai/edit/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          generation_id: generationId, // Use saved generation ID
          start_frame: startFrame,
          end_frame: endFrame,
          interval: interval,
        }),
      }).then((res) => {
        if (!res.ok) {
          const errorData = res.json().catch(() => ({}));
          throw new Error(`Accept failed: ${res.status}`);
        }
        return res.json();
      }).then(async (data) => {
        if (data.error) {
          // Special case: if edit is already in progress, don't reset to idle
          // Instead, sync with current backend state and keep applying status
          if (data.error === "Edit already in progress" && data.status === "processing") {
            console.log("Edit already in progress - syncing state with backend progress");
            // Sync state with backend's current progress
            setState((prev) => ({
              ...prev,
              aiEditStatus: "applying",
              aiEditProgress: data.progress || prev.aiEditProgress,
              aiEditPhase: data.phase || prev.aiEditPhase,
              aiInterpolationProgress: data.interpolation_progress || prev.aiInterpolationProgress,
            }));
            // Don't reset ref - keep it as true so loading screen shows
            // Polling will continue to update the status
            return;
          }
          
          console.error("Accept error from backend:", data.error);
          acceptInProgressRef.current = false; // Reset ref on error
          setState((prev) => ({
            ...prev,
            aiEditStatus: "idle",
            aiEditProgress: { done: 0, total: 0 },
            aiEditPhase: null,
            aiInterpolationProgress: { done: 0, total: 0 },
            showToast: true,
            toastMessage: data.error,
          }));
          return;
        }
        console.log("Accept started successfully, status:", data);
        // Keep status as "applying" - unified polling will handle updates
        // Don't reset status here, let polling update it
      }).catch((err) => {
        acceptInProgressRef.current = false; // Reset ref on error
        console.error("Accept error:", err);
        setState((prev) => ({
          ...prev,
          aiEditStatus: "idle",
          aiEditProgress: { done: 0, total: 0 },
          aiEditPhase: null,
          aiInterpolationProgress: { done: 0, total: 0 },
          showToast: true,
          toastMessage: `Failed to accept: ${err.message}`,
        }));
      });

      return s;
    });
  }, []);

  const rejectAIGeneration = useCallback(() => {
    setState((s) => {
      if (!s.projectId || !s.aiGenerationId) return s;

      fetch(`${API_URL}/ai/edit/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          generation_id: s.aiGenerationId,
        }),
      });

      return {
        ...s,
        aiPreviewFrameUrl: null,
        aiGenerationId: null,
        aiEditStatus: "idle",
      };
    });
  }, []);

  const retryAIGeneration = useCallback(() => {
    setState((s) => {
      if (!s.projectId || !s.aiGenerationId) return s;

      setState((prev) => ({
        ...prev,
        isAIGenerating: true,
      }));

      fetch(`${API_URL}/ai/edit/retry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: s.projectId,
          generation_id: s.aiGenerationId,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          const assistantMessage: ChatMessage = {
            role: "assistant",
            message: "Preview regenerated! Review it below.",
            timestamp: Date.now(),
          };
          setState((prev) => ({
            ...prev,
            aiChatHistory: [...prev.aiChatHistory, assistantMessage],
            aiPreviewFrameUrl: `${API_URL}${data.preview_url}`,
            aiGenerationId: data.generation_id,
            isAIGenerating: false,
            aiEditStatus: "preview",
          }));
        })
        .catch((err) => {
          console.error("AI retry error:", err);
          setState((prev) => ({
            ...prev,
            isAIGenerating: false,
            showToast: true,
            toastMessage: "Failed to retry generation",
          }));
        });

      return s;
    });
  }, []);

  const selectedObject = useMemo(
    () => state.detections.find((d) => d.id === state.selectedObjectId) ?? null,
    [state.detections, state.selectedObjectId]
  );

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
    refineFrame,
    propagateEdit,
    undoEdit,
    setCurrentFrame,
    togglePlay,
    setZoom,
    setApplyToAllFrames,
    setEditRange,
    closeEditPanel,
    setVideoName,
    hideToast,
    sendAIPrompt,
    acceptAIGeneration,
    rejectAIGeneration,
    retryAIGeneration,
    editProgress: state.editProgress,
    editStatus: state.editStatus,
    changeMarkers: state.changeMarkers,
    handleMarkerDrag,
  };
}
