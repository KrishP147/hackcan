"use client";

import { useRouter } from "next/navigation";
import { Play, Trash2, Clock } from "lucide-react";
import { useEffect, useState } from "react";
import type { Project } from "@/lib/supabase";

export function VideoHistory() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);

  useEffect(() => {
    let active = true;

    void fetch("/api/projects", { cache: "no-store" })
      .then(async (response) => {
        // Guests deliberately see no history. The API also filters signed-in
        // results by the Auth0 subject, so projects cannot leak across users.
        if (!response.ok) return [];
        const data = await response.json();
        return Array.isArray(data) ? data : [];
      })
      .then((data: Project[]) => {
        if (active) setProjects(data);
      })
      .catch(() => {
        if (active) setProjects([]);
      });

    return () => {
      active = false;
    };
  }, []);

  const formatDate = (timestamp: string) => {
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

  if (projects.length === 0) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <h2 className="text-2xl font-bold text-[var(--fg)] mb-6">Recent Videos</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {projects.map((project) => (
          <div
            key={project.project_id}
            className="bg-[var(--surface-dark)] rounded-xl p-4 border border-[var(--border)] hover:border-[var(--accent)] transition-all duration-300 group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-white truncate mb-1">
                  {project.name || "Untitled Project"}
                </h3>
                <div className="flex items-center gap-2 text-sm text-white/55">
                  <Clock className="w-3 h-3" />
                  <span>{formatDate(project.updated_at)}</span>
                </div>
                {Boolean(project.frame_count) && (
                  <div className="text-xs text-white/45 mt-1">
                    {project.frame_count} frames
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  router.push(`/editor/${project.project_id}?frame=${project.last_frame || 0}`);
                }}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent)] text-white font-medium hover:bg-[var(--accent-hover)] transition-colors"
              >
                <Play className="w-4 h-4" />
                Open
              </button>
              <button
                onClick={async (e) => {
                  e.stopPropagation();
                  if (!confirm(`Delete "${project.name || "Untitled Project"}"?`)) return;
                  const response = await fetch(`/api/projects/${project.project_id}`, {
                    method: "DELETE",
                  });
                  if (response.ok) {
                    setProjects((current) => current.filter(
                      (item) => item.project_id !== project.project_id,
                    ));
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
