"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Play, Loader2 } from "lucide-react";
import { useVideoStore } from "@/stores/videoStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function Footer() {
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const addProject = useVideoStore((state) => state.addProject);
  const setCurrentProject = useVideoStore((state) => state.setCurrentProject);

  async function uploadFile(file: File) {
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
    const data = await res.json();
    
    // Save to Zustand store for persistence
    addProject({
      projectId: data.project_id,
      videoName: file.name,
      uploadedAt: Date.now(),
      status: "created",
    });
    setCurrentProject(data.project_id);
    
    router.push(`/editor/${data.project_id}`);
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    if (e.target) e.target.value = "";
  };

  return (
    <footer className="bg-[var(--surface-dark)] py-12 px-6 md:px-24">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
            <Play className="w-3.5 h-3.5 text-white ml-0.5" fill="white" />
          </div>
          <span className="text-lg font-[550] text-white tracking-tight">
            ProductName
          </span>
        </div>

        {/* CTA */}
        <button
          onClick={() => fileInputRef.current?.click()}
          className="inline-flex items-center gap-2 px-7 py-3 rounded-xl bg-[var(--accent)] text-white font-semibold transition-all duration-300 hover:bg-[var(--accent-hover)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer disabled:opacity-50"
          disabled={uploading}
        >
          {uploading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Uploading…
            </>
          ) : (
            "Get Started"
          )}
        </button>
      </div>

      <div className="max-w-7xl mx-auto mt-8 pt-6 border-t border-white/10">
        <p className="text-sm text-[var(--fg-subtle)] text-center md:text-left">
          &copy; 2026 ProductName. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
