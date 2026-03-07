"use client";

import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    // Future: handle file
  }, []);

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
        <button className="inline-flex items-center gap-2 px-7 py-3 rounded-xl bg-[var(--accent)] text-white font-semibold transition-all duration-300 hover:bg-[var(--accent-hover)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          Get Started
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
      />
    </div>
  );
}
