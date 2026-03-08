"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, Loader2 } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function DropZone() {
  const router = useRouter();
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function uploadFile(file: File) {
    setUploading(true);
    setStatus("Uploading video...");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
    const data = await res.json();

    // Derive thumbnail from Cloudinary video URL (snapshot at 0s)
    const thumbnailUrl = data.video_url
      ? data.video_url
          .replace("/video/upload/", "/video/upload/so_0,w_640/")
          .replace(/\.(mp4|mov|webm|avi)$/i, ".jpg")
      : null;

    // Register project in Supabase (best-effort — don't block navigation on failure)
    fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: data.project_id,
        name: file.name,
        thumbnail_url: thumbnailUrl,
      }),
    }).catch(() => {});

    // Redirect immediately — editor will kick off extract and poll for readiness
    router.push(`/editor/${data.project_id}`);
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
      className={`max-w-2xl mx-auto rounded-2xl p-12 text-center transition-all duration-300 border-2 ${
        isDragging
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
