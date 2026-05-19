"use client";

import { EditorTopBar } from "@/components/editor/EditorTopBar";
import { EditorCanvas } from "@/components/editor/EditorCanvas";
import { EditorTimeline } from "@/components/editor/EditorTimeline";
import { EditToolbar } from "@/components/editor/EditToolbar";
import { AIChatPane } from "@/components/editor/AIChatPane";
import { Toast } from "@/components/editor/Toast";
import { AIProgressOverlay } from "@/components/editor/AIProgressOverlay";
import { EditProgressOverlay } from "@/components/editor/EditProgressOverlay";
import { useEditorState } from "@/hooks/useEditorState";
import { useProjectSync } from "@/hooks/useProjectSync";
import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import { useVideoStore } from "@/stores/videoStore";

export default function EditorPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const projectId = params.projectId as string;
  const initialFrame = Number(searchParams.get("frame") ?? 0);
  const setCurrentProject = useVideoStore((state) => state.setCurrentProject);
  const editor = useEditorState(projectId, initialFrame);

  useProjectSync({
    projectId,
    currentFrame: editor.currentFrame,
    videoLoaded: editor.videoLoaded,
    status: editor.videoLoaded ? "ready" : "created",
    thumbnailUrl: editor.storageBaseUrl ? `${editor.storageBaseUrl}/frame_0001.jpg` : null,
  });

  // Set current project in Zustand when page loads
  useEffect(() => {
    if (projectId) {
      setCurrentProject(projectId);
    }
  }, [projectId, setCurrentProject]);
  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const currentFrameRef = useRef(editor.currentFrame);
  const framesLengthRef = useRef(editor.frames.length);
  const [isDark, setIsDark] = useState(false);

  // Playback loop — 15 fps so frames have time to load
  const playbackFps = 15;
  useEffect(() => {
    if (editor.isPlaying && editor.videoLoaded) {
      playIntervalRef.current = setInterval(() => {
        editor.setCurrentFrame(
          editor.currentFrame >= editor.frames.length - 1
            ? 0
            : editor.currentFrame + 1
        );
      }, 1000 / playbackFps);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [editor.isPlaying, editor.videoLoaded, editor.currentFrame, editor.frames.length, editor.setCurrentFrame]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case " ":
          e.preventDefault();
          editor.togglePlay();
          break;
        case "ArrowLeft":
          editor.setCurrentFrame(Math.max(0, editor.currentFrame - 1));
          break;
        case "ArrowRight":
          editor.setCurrentFrame(
            Math.min(editor.frames.length - 1, editor.currentFrame + 1)
          );
          break;
        case "Escape":
          editor.closeEditPanel();
          break;
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [editor]);

  return (
    <div
      className={`h-screen flex flex-col overflow-hidden select-none transition-colors duration-300 ${isDark ? "editor-dark" : "editor-light"}`}
      style={{ background: "var(--ed-bg)" }}
    >
      <EditorTopBar
        videoName={editor.videoName}
        onNameChange={editor.setVideoName}
        videoLoaded={editor.videoLoaded}
        isDark={isDark}
        onToggleTheme={() => setIsDark((d) => !d)}
        editApplied={editor.editVersion > 0}
        onUndo={editor.undoEdit}
        onSave={editor.exportToMp4}
        isExporting={editor.isExporting}
      />

      <div className="flex-1 flex overflow-hidden">
        <EditorCanvas
          projectId={projectId}
          videoLoaded={editor.videoLoaded}
          detections={editor.detections}
          isDetecting={editor.isDetecting}
          isSegmenting={editor.isSegmenting}
          maskCount={editor.maskCount}
          maskVersion={editor.maskVersion}
          editVersion={editor.editVersion}
          transformedFrameVersions={editor.transformedFrameVersions}
          selectedObjectId={editor.selectedObjectId}
          editMode={editor.editMode}
          editParams={editor.editParams}
          isProcessing={editor.isProcessing}
          zoom={editor.zoom}
          currentFrame={editor.currentFrame}
          totalFrames={editor.frames.length}
          frameWidth={editor.frameWidth}
          frameHeight={editor.frameHeight}
          previewFrameUrl={editor.aiPreviewFrameUrl}
          aiEditStatus={editor.aiEditStatus}
          storageBaseUrl={editor.storageBaseUrl}
          onSelectObject={editor.selectObject}
          onUpload={editor.loadVideo}
          onApplyEdit={editor.applyEditAction}
          onSegmentAtPoint={editor.segmentAtPoint}
          onCancelEdit={editor.cancelEdit}
        />

        <EditToolbar
          objectLabel={
            editor.selectedObjectId
              ? editor.detections.find((d) => d.id === editor.selectedObjectId)?.label || "object"
              : "selection"
          }
          active={!editor.isSegmenting && editor.videoLoaded}
          hasMask={editor.maskCount > 0}
          editApplied={editor.editVersion > 0}
          onApply={editor.applyEditAction}
          onRefine={editor.refineFrame}
          onPropagate={editor.propagateEdit}
          onUndo={editor.undoEdit}
          onClose={editor.closeEditPanel}
          isRefining={editor.isRefining}
        />

        <AIChatPane
          projectId={projectId}
          currentFrame={editor.currentFrame}
          videoLoaded={editor.videoLoaded}
          chatHistory={editor.aiChatHistory}
          previewFrameUrl={editor.aiPreviewFrameUrl}
          isGenerating={editor.isAIGenerating}
          aiEditStatus={editor.aiEditStatus}
          onSendPrompt={editor.sendAIPrompt}
          onAccept={editor.acceptAIGeneration}
          onReject={editor.rejectAIGeneration}
          onRetry={editor.retryAIGeneration}
        />
      </div>

      <EditorTimeline
        videoLoaded={editor.videoLoaded}
        currentFrame={editor.currentFrame}
        totalFrames={editor.frames.length}
        duration={editor.duration}
        fps={editor.fps}
        isPlaying={editor.isPlaying}
        isProcessing={editor.isProcessing}
        zoom={editor.zoom}
        editRangeStart={editor.editRangeStart}
        editRangeEnd={editor.editRangeEnd}
        changeMarkers={editor.changeMarkers}
        onFrameChange={editor.setCurrentFrame}
        onTogglePlay={editor.togglePlay}
        onZoomChange={editor.setZoom}
        onEditRangeChange={editor.setEditRange}
        onMarkerDrag={editor.handleMarkerDrag}
      />

      <Toast
        message={editor.toastMessage}
        show={editor.showToast}
        onHide={editor.hideToast}
      />

      <AIProgressOverlay
        show={editor.aiEditStatus === "applying"}
        progress={editor.aiEditProgress}
        interpolationProgress={editor.aiInterpolationProgress}
        phase={editor.aiEditPhase}
        status={editor.aiEditStatus}
        onCancel={editor.cancelEdit}
      />

      <EditProgressOverlay
        show={editor.isProcessing && (editor.editStatus === "uploading" || editor.editStatus === "editing")}
        progress={editor.editProgress}
        status={editor.editStatus}
      />
    </div>
  );
}
