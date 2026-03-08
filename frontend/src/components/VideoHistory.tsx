"use client";

import { useRouter } from "next/navigation";
import { useVideoStore } from "@/stores/videoStore";
import { Play, Trash2, Clock } from "lucide-react";
import { useMemo } from "react";

export function VideoHistory() {
  const router = useRouter();
  const projects = useVideoStore((state) => state.projects);
  const removeProject = useVideoStore((state) => state.removeProject);
  const setCurrentProject = useVideoStore((state) => state.setCurrentProject);

  // Sort by most recent first
  const sortedProjects = useMemo(() => {
    return [...projects].sort((a, b) => b.uploadedAt - a.uploadedAt);
  }, [projects]);

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  if (sortedProjects.length === 0) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Recent Videos</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedProjects.map((project) => (
          <div
            key={project.projectId}
            className="bg-[var(--surface-dark)] rounded-xl p-4 border border-[var(--border)] hover:border-[var(--accent)] transition-all duration-300 group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-[var(--fg)] truncate mb-1">
                  {project.videoName}
                </h3>
                <div className="flex items-center gap-2 text-sm text-[var(--fg-muted)]">
                  <Clock className="w-3 h-3" />
                  <span>{formatDate(project.uploadedAt)}</span>
                </div>
                {project.frameCount && (
                  <div className="text-xs text-[var(--fg-subtle)] mt-1">
                    {project.frameCount} frames
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setCurrentProject(project.projectId);
                  router.push(`/editor/${project.projectId}`);
                }}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] text-white font-medium hover:bg-[var(--accent-hover)] transition-colors"
              >
                <Play className="w-4 h-4" />
                Open
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Delete "${project.videoName}"?`)) {
                    removeProject(project.projectId);
                  }
                }}
                className="p-2 rounded-lg text-[var(--fg-muted)] hover:text-[var(--accent)] hover:bg-[var(--bg-subtle)] transition-colors"
                title="Delete"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
