"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { Project } from "@/lib/supabase";

const STATUS_STYLES: Record<string, { label: string; bg: string; text: string }> = {
  created:    { label: "Created",    bg: "#F3F4F6", text: "#6B7280" },
  extracting: { label: "Extracting", bg: "#FEF3C7", text: "#D97706" },
  detecting:  { label: "Detecting",  bg: "#FEF3C7", text: "#D97706" },
  ready:      { label: "Ready",      bg: "#D1FAE5", text: "#059669" },
  processing: { label: "Processing", bg: "#DBEAFE", text: "#2563EB" },
  done:       { label: "Done",       bg: "#D1FAE5", text: "#059669" },
  error:      { label: "Error",      bg: "#FEE2E2", text: "#DC2626" },
};

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  });
}

export function ProjectCard({ project, onDelete }: { project: Project; onDelete: (id: string) => void }) {
  const router = useRouter();
  const [deleting, setDeleting] = useState(false);
  const [thumbnailUrl, setThumbnailUrl] = useState(project.thumbnail_url ?? "");
  const [settingThumbnail, setSettingThumbnail] = useState(false);
  const status = STATUS_STYLES[project.status] ?? STATUS_STYLES.created;

  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm("Delete this project?")) return;
    setDeleting(true);
    await fetch(`/api/projects/${project.project_id}`, { method: "DELETE" });
    onDelete(project.project_id);
  }

  async function handleSetThumbnail(e: React.MouseEvent) {
    e.stopPropagation();
    if (settingThumbnail) return;
    setSettingThumbnail(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/project/${project.project_id}/status`);
      if (!res.ok) return;
      const { frame_count, status: projectStatus } = await res.json();
      if (!frame_count || !["ready", "done"].includes(projectStatus)) return;
      const url = `${process.env.NEXT_PUBLIC_API_URL}/frame/${project.project_id}/${Math.floor(Math.random() * frame_count)}`;
      setThumbnailUrl(url);
      await fetch(`/api/projects/${project.project_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thumbnail_url: url }),
      });
    } catch {
      // silently fail
    } finally {
      setSettingThumbnail(false);
    }
  }

  return (
    <div className="group relative bg-white border border-[#E5E7EB] rounded-2xl overflow-hidden hover:border-[#F43F5E] hover:shadow-md transition-all duration-300 hover:-translate-y-1 cursor-pointer">
      {/* Thumbnail */}
      <div
        className="aspect-video bg-[#111827] flex items-center justify-center relative"
        onClick={() => router.push(`/editor/${project.project_id}?frame=${project.last_frame}`)}
      >
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt={project.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#374151" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="2" width="20" height="20" rx="3" />
            <polygon points="10 8 16 12 10 16 10 8" fill="#374151" stroke="none" />
          </svg>
        )}
        {/* Resume overlay */}
        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          <span className="bg-[#F43F5E] text-white text-sm font-semibold px-4 py-2 rounded-xl">
            Resume
          </span>
        </div>
      </div>

      {/* Info */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <p className="font-[550] text-[#171717] text-base leading-tight line-clamp-1 flex-1">
            {project.name}
          </p>
          <button
            onClick={handleSetThumbnail}
            disabled={settingThumbnail}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-[#9CA3AF] hover:text-[#F43F5E] text-xs p-1 -mt-0.5 flex-shrink-0"
            title="Set random thumbnail"
          >
            {settingThumbnail ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
                <path d="M21 12a9 9 0 11-6.219-8.56" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            )}
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-[#9CA3AF] hover:text-[#DC2626] text-xs p-1 -mt-0.5 flex-shrink-0"
            title="Delete project"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
              <path d="M10 11v6M14 11v6" />
              <path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2" />
            </svg>
          </button>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[#9CA3AF] text-xs">{formatDate(project.updated_at)}</span>
          <span
            className="text-xs font-semibold px-2 py-0.5 rounded-full"
            style={{ background: status.bg, color: status.text }}
          >
            {status.label}
          </span>
        </div>
      </div>
    </div>
  );
}
