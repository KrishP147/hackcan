"use client";

import { EditorTopBar } from "@/components/editor/EditorTopBar";
import { EditorCanvas } from "@/components/editor/EditorCanvas";
import { EditorTimeline } from "@/components/editor/EditorTimeline";
import { EditToolbar } from "@/components/editor/EditToolbar";
import { Toast } from "@/components/editor/Toast";
import { useEditorState } from "@/hooks/useEditorState";
import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";

export default function EditorPage() {
  const params = useParams();
  const projectId = params.projectId as string;
  const editor = useEditorState(projectId);
  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [isDark, setIsDark] = useState(false);

  // Playback loop
  useEffect(() => {
    if (editor.isPlaying && editor.videoLoaded) {
      playIntervalRef.current = setInterval(() => {
        editor.setCurrentFrame(
          editor.currentFrame >= editor.frames.length - 1
            ? 0
            : editor.currentFrame + 1
        );
      }, 1000 / editor.fps);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [editor.isPlaying, editor.videoLoaded, editor.currentFrame, editor.fps, editor.frames.length, editor.setCurrentFrame]);

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
          selectedObjectId={editor.selectedObjectId}
          editMode={editor.editMode}
          editParams={editor.editParams}
          isProcessing={editor.isProcessing}
          zoom={editor.zoom}
          currentFrame={editor.currentFrame}
          totalFrames={editor.frames.length}
          frameWidth={editor.frameWidth}
          frameHeight={editor.frameHeight}
          onSelectObject={editor.selectObject}
          onUpload={editor.loadVideo}
          onApplyEdit={editor.applyEditAction}
          onSegmentAtPoint={editor.segmentAtPoint}
        />

        <EditToolbar
          objectLabel={
            editor.selectedObjectId
              ? editor.detections.find((d) => d.id === editor.selectedObjectId)?.label || "object"
              : "selection"
          }
          active={!editor.isSegmenting && editor.maskCount > 0}
          onApply={editor.applyEditAction}
          onClose={editor.closeEditPanel}
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
        onFrameChange={editor.setCurrentFrame}
        onTogglePlay={editor.togglePlay}
        onZoomChange={editor.setZoom}
        onEditRangeChange={editor.setEditRange}
      />

      <Toast
        message={editor.toastMessage}
        show={editor.showToast}
        onHide={editor.hideToast}
      />
    </div>
  );
}
