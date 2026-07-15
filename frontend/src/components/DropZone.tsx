"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, Clock3, Upload, Loader2 } from "lucide-react";
import { useVideoStore } from "@/stores/videoStore";
import { uploadProjectVideo } from "@/lib/upload-project";
import {
  MAX_VIDEO_DURATION_SECONDS,
} from "@/lib/video-upload";

export function DropZone() {
  const router = useRouter();
  const addProject = useVideoStore((state) => state.addProject);
  const setCurrentProject = useVideoStore((state) => state.setCurrentProject);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function uploadFile(file: File) {
    setUploading(true);
    setStatus("Checking video length...");

    try {
      const data = await uploadProjectVideo(file, setStatus);

      // Keep guest history in this browser. Signed-in users additionally get a
      // durable Supabase row that appears on every device in /dashboard.
      addProject({
        projectId: data.project_id,
        videoName: file.name,
        uploadedAt: Date.now(),
        status: data.compute_available ? "processing" : "stored",
        storagePath: data.storage_path,
      });
      setCurrentProject(data.project_id);

      // The editor shows the durable video while Modal warms its working cache.
      router.push(`/editor/${data.project_id}`);
    } catch (error) {
      setUploading(false);
      setStatus(error instanceof Error ? error.message : "Upload failed");
    }
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("video/")) {
      uploadFile(file);
    } else {
      setStatus("Please choose a video file.");
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    if (e.target) e.target.value = "";
  };

  const handleUploadClick = () => fileInputRef.current?.click();

  return (
    <div
      data-dropzone
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`max-w-2xl mx-auto rounded-2xl p-12 text-center transition-all duration-300 border-2 ${isDragging
          ? "border-solid border-[var(--accent)] bg-[rgba(244,63,94,0.04)] animate-pulse-border"
          : "border-dashed border-[var(--border)]"
        }`}
    >
      {uploading ? (
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-8 h-8 text-[var(--accent)] animate-spin" />
          <p className="text-[var(--fg-muted)] text-lg">{status}</p>
        </div>
      ) : (
        <>
          <p className="text-[var(--fg-muted)] text-lg mb-4">
            Drag and drop your video here
          </p>
          <div className="mb-5 flex items-center justify-center gap-2 text-sm font-semibold text-[var(--accent)]">
            <Clock3 className="h-4 w-4" />
            Videos must be under {MAX_VIDEO_DURATION_SECONDS} seconds
          </div>
          <p className="text-[var(--fg-subtle)] text-sm mb-6">or</p>
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <button
              onClick={handleUploadClick}
              className="inline-flex items-center gap-2 px-7 py-3 rounded-xl border border-[var(--fg)] text-[var(--fg)] font-semibold transition-all duration-300 hover:bg-[var(--bg-subtle)] hover:border-[var(--accent)] active:scale-[0.98] cursor-pointer"
            >
              <Upload className="w-4 h-4" />
              Upload from device
            </button>
          </div>
          {status && (
            <p
              role="alert"
              className="mt-5 flex items-center justify-center gap-2 text-sm font-medium text-red-500"
            >
              <AlertCircle className="h-4 w-4 shrink-0" />
              {status}
            </p>
          )}
        </>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
