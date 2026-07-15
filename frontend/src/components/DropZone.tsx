"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, Loader2, Play } from "lucide-react";
import { useVideoStore } from "@/stores/videoStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function DropZone() {
  const router = useRouter();
  const addProject = useVideoStore((state) => state.addProject);
  const setCurrentProject = useVideoStore((state) => state.setCurrentProject);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  function registerProject(projectId: string, name: string) {
    return fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        name,
        thumbnail_url: null,
      }),
    }).catch(() => { });
  }

  async function uploadFile(file: File) {
    setUploading(true);
    setStatus("Uploading video...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || data.error || !data.project_id) {
        throw new Error(data.error || "Upload failed");
      }

      // Register project in Supabase (best-effort — don't block navigation on failure)
      await registerProject(data.project_id, file.name);

      // Redirect immediately — editor will kick off extract and poll for readiness
      router.push(`/editor/${data.project_id}`);
    } catch (error) {
      setUploading(false);
      setStatus(error instanceof Error ? error.message : "Upload failed");
    }
  }

  async function loadDemoVideo() {
    setUploading(true);
    setStatus("Preparing the FrameShift demo...");
    try {
      const response = await fetch(`${API_URL}/demo`, { method: "POST" });
      const data = await response.json();
      if (!response.ok || data.error || !data.project_id) {
        throw new Error(data.detail || data.error || "Demo video is unavailable");
      }
      await registerProject(data.project_id, data.video_name || "FrameShift-demo.mp4");
      router.push(`/editor/${data.project_id}`);
    } catch (error) {
      setUploading(false);
      setStatus(error instanceof Error ? error.message : "Could not load demo video");
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
          <p className="text-[var(--fg-subtle)] text-sm mb-6">or</p>
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <button
              onClick={handleUploadClick}
              className="inline-flex items-center gap-2 px-7 py-3 rounded-xl border border-[var(--fg)] text-[var(--fg)] font-semibold transition-all duration-300 hover:bg-[var(--bg-subtle)] hover:border-[var(--accent)] active:scale-[0.98] cursor-pointer"
            >
              <Upload className="w-4 h-4" />
              Upload from device
            </button>
            <button
              onClick={handleUploadClick}
              className="inline-flex items-center gap-2 px-7 py-3 rounded-xl bg-[var(--accent)] text-white font-semibold transition-all duration-300 hover:bg-[var(--accent-hover)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
            >
              Get Started
            </button>
          </div>
          <button
            onClick={loadDemoVideo}
            className="inline-flex items-center gap-2 mt-5 text-sm font-semibold text-[var(--accent)] transition-opacity hover:opacity-70"
          >
            <Play className="w-3.5 h-3.5" fill="currentColor" />
            Try the built-in demo video
          </button>
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
