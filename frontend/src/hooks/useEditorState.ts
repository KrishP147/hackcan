"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type Detection,
  type EditMode,
  type EditParams,
  type FrameData,
  generateFrames,
} from "@/lib/mock-data";

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
  selectedObjectId: string | null;
  editMode: EditMode | null;
  editParams: EditParams;
  isProcessing: boolean;
  applyToAllFrames: boolean;
  editRangeStart: number;
  editRangeEnd: number;
  zoom: number;
  showEditPanel: boolean;
  toastMessage: string;
  showToast: boolean;
}

const DEFAULT_EDIT_PARAMS: EditParams = {
  recolor: { color: "#F43F5E", opacity: 0.6 },
  resize: { scale: 1.0 },
  replace: { imageUrl: null },
};

export function useEditorState(projectId?: string) {
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
    applyToAllFrames: true,
    editRangeStart: 0,
    editRangeEnd: 0,
    zoom: 100,
    showEditPanel: false,
    toastMessage: "",
    showToast: false,
  });

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll backend for project status when projectId is provided
  useEffect(() => {
    if (!projectId) return;

    const extractTriggered = { current: false };

    const poll = async () => {
      try {
        const res = await fetch(`${API_URL}/project/${projectId}/status`);
        const status = await res.json();

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
            setState((s) => ({
              ...s,
              projectId,
              videoLoaded: true,
              videoName: projectId,
              fps: 30,
              duration: frameCount / 30,
              frames: generateFrames(frameCount),
              frameWidth: status.frame_width || 0,
              frameHeight: status.frame_height || 0,
              editRangeEnd: s.editRangeEnd === 0 ? frameCount - 1 : s.editRangeEnd,
              isDetecting: !!status.detecting,
              isSegmenting: !!status.segmenting,
              maskCount: status.mask_count || 0,
            }));

            // Store all per-frame detections
            if (status.detections && Object.keys(status.detections).length > 0) {
              setState((s) => ({ ...s, allDetections: status.detections }));
            }

            // Stop polling once ready, not detecting, and not segmenting
            if (status.status === "ready" && !status.detecting && !status.segmenting) {
              if (pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
              }
            }
          }
        }
      } catch {
        // Backend not reachable yet, keep polling
      }
    };

    poll();
    pollingRef.current = setInterval(poll, 1500);

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
      }).then(() => {
        // Clear any existing polling, then start fresh
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
        const poll = async () => {
          try {
            const res = await fetch(`${API_URL}/project/${s.projectId}/status`);
            const status = await res.json();
            setState((prev) => ({
              ...prev,
              isSegmenting: !!status.segmenting,
              maskCount: status.mask_count || 0,
              maskVersion: prev.maskVersion + ((status.mask_count || 0) > 0 && !status.segmenting ? 1 : 0),
            }));
            if (!status.segmenting && status.segment_status === "done") {
              if (pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
              }
            }
          } catch { /* keep polling */ }
        };
        pollingRef.current = setInterval(poll, 1500);
      });

      return {
        ...s,
        isSegmenting: true,
        maskCount: 0,
        selectedObjectId: null,
        showEditPanel: false,
      };
    });
  }, []);

  const selectObject = useCallback((id: string | null) => {
    setState((s) => {
      // Trigger segmentation when selecting an object
      if (id !== null && s.projectId && s.frameWidth > 0) {
        const det = s.detections.find((d) => d.id === id);
        if (det) {
          const [xPct, yPct, wPct, hPct] = det.bbox;
          const clickX = Math.round(((xPct + wPct / 2) / 100) * s.frameWidth);
          const clickY = Math.round(((yPct + hPct / 2) / 100) * s.frameHeight);

          fetch(`${API_URL}/segment`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              project_id: s.projectId,
              frame_index: s.currentFrame + 1,
              click_x: clickX,
              click_y: clickY,
            }),
          }).then(() => {
            // Clear any existing polling, then start fresh
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
            const poll = async () => {
              try {
                const res = await fetch(`${API_URL}/project/${s.projectId}/status`);
                const status = await res.json();
                setState((prev) => ({
                  ...prev,
                  isSegmenting: !!status.segmenting,
                  maskCount: status.mask_count || 0,
                  maskVersion: prev.maskVersion + ((status.mask_count || 0) > 0 && !status.segmenting ? 1 : 0),
                }));
                if (!status.segmenting && status.segment_status === "done") {
                  if (pollingRef.current) {
                    clearInterval(pollingRef.current);
                    pollingRef.current = null;
                  }
                }
              } catch { /* ignore */ }
            };
            pollingRef.current = setInterval(poll, 1500);
          });
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

  const applyEditAction = useCallback(
    (action: string, params: { color?: string; prompt?: string; scale?: number }) => {
      setState((s) => {
        if (!s.projectId) return s;

        // Use edit range, falling back to full video if range hasn't been set
        const startFrame = s.editRangeStart + 1;  // 1-based for backend
        const endFrame = s.editRangeEnd > 0 ? s.editRangeEnd + 1 : s.frames.length;
        const editRule: Record<string, unknown> = {
          edit_type: action,
          start_frame: startFrame,
          end_frame: endFrame,
        };
        if (params.color) editRule.color = params.color;
        if (params.prompt) editRule.prompt = params.prompt;
        if (params.scale) editRule.scale = params.scale;

        fetch(`${API_URL}/edit`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project_id: s.projectId,
            edit_rules: [editRule],
          }),
        }).then(() => {
          // Clear any existing polling, then poll for edit completion
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          const pollEdit = async () => {
            try {
              const res = await fetch(`${API_URL}/project/${s.projectId}/status`);
              const status = await res.json();
              if (status.edit_status === "done" || status.edit_status === "error") {
                setState((prev) => ({
                  ...prev,
                  isProcessing: false,
                  editVersion: prev.editVersion + 1,
                  showToast: true,
                  toastMessage: status.edit_status === "done"
                    ? `${action} applied successfully`
                    : `Edit failed: ${status.edit_error || "unknown error"}`,
                }));
                if (pollingRef.current) {
                  clearInterval(pollingRef.current);
                  pollingRef.current = null;
                }
              }
            } catch { /* keep polling */ }
          };
          pollingRef.current = setInterval(pollEdit, 1500);
        });

        return { ...s, isProcessing: true, selectedObjectId: null, showEditPanel: false };
      });
    },
    []
  );

  const setCurrentFrame = useCallback((frame: number) => {
    setState((s) => ({ ...s, currentFrame: frame }));
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
    applyEditAction,
    setCurrentFrame,
    togglePlay,
    setZoom,
    setApplyToAllFrames,
    setEditRange,
    closeEditPanel,
    setVideoName,
    hideToast,
  };
}
