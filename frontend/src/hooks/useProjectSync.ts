"use client";

import { useCallback, useEffect, useRef } from "react";

type ProjectUpdates = {
  last_frame?: number;
  status?: string;
  thumbnail_url?: string;
};

/**
 * Persists editor metadata for signed-in users. The API route owns the auth
 * decision, so this remains reliable while the Auth0 client session hydrates.
 * Guests receive a harmless 401 and can still edit normally.
 */
export function useProjectSync({
  projectId,
  currentFrame,
  videoLoaded,
  status,
  thumbnailUrl,
  name,
}: {
  projectId: string;
  currentFrame: number;
  videoLoaded: boolean;
  status: string;
  thumbnailUrl?: string | null;
  name?: string;
}) {
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const currentFrameRef = useRef(currentFrame);
  const lastSyncedFrame = useRef<number>(-1);
  const lastSyncedStatus = useRef<string>("");
  const thumbnailSynced = useRef(false);

  useEffect(() => {
    lastSyncedFrame.current = -1;
    lastSyncedStatus.current = "";
    thumbnailSynced.current = false;
  }, [projectId]);

  useEffect(() => {
    currentFrameRef.current = currentFrame;
  }, [currentFrame]);

  const registerProject = useCallback(async () => {
    if (!projectId) return false;

    const payload: Record<string, string> = { project_id: projectId };
    const trimmedName = name?.trim();
    // useEditorState temporarily falls back to the project id while account
    // metadata loads. Never persist that fallback as the project title.
    if (trimmedName && trimmedName !== projectId) payload.name = trimmedName;
    if (thumbnailUrl) payload.thumbnail_url = thumbnailUrl;

    try {
      const response = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok && response.status !== 401) {
        console.warn("Project registration failed", response.status, await response.text());
      }
      return response.ok;
    } catch (error) {
      console.warn("Project registration failed", error);
      return false;
    }
  }, [name, projectId, thumbnailUrl]);

  const patchProject = useCallback(async (updates: ProjectUpdates) => {
    if (!projectId || !(await registerProject())) return false;
    try {
      const response = await fetch(`/api/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      if (!response.ok && response.status !== 401) {
        console.warn("Project sync failed", response.status, await response.text());
      }
      return response.ok;
    } catch (error) {
      console.warn("Project sync failed", error);
      return false;
    }
  }, [projectId, registerProject]);

  // Idempotently register the project. This also claims a project that was
  // uploaded as a guest immediately before the user chose to sign in.
  useEffect(() => {
    void registerProject();
  }, [registerProject]);

  // Debounced frame position sync
  useEffect(() => {
    if (!projectId || !videoLoaded) return;
    if (currentFrame === lastSyncedFrame.current) return;

    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      void patchProject({ last_frame: currentFrame }).then((saved) => {
        if (saved) lastSyncedFrame.current = currentFrame;
      });
    }, 1000);

    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [currentFrame, patchProject, projectId, videoLoaded]);

  // A debounce should not lose the final playhead position when the user
  // leaves the editor. Registration has already started above, so this direct
  // keepalive request can finish while the page is navigating away.
  useEffect(() => {
    if (!projectId || !videoLoaded) return;

    const flushFrame = () => {
      const frame = currentFrameRef.current;
      if (frame === lastSyncedFrame.current) return;
      lastSyncedFrame.current = frame;
      void fetch(`/api/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ last_frame: frame }),
        keepalive: true,
      }).catch(() => {});
    };

    window.addEventListener("pagehide", flushFrame);
    return () => {
      window.removeEventListener("pagehide", flushFrame);
      flushFrame();
    };
  }, [projectId, videoLoaded]);

  // Sync status changes to Supabase
  useEffect(() => {
    if (!projectId || !status) return;
    if (status === lastSyncedStatus.current) return;
    void patchProject({ status }).then((saved) => {
      if (saved) lastSyncedStatus.current = status;
    });
  }, [patchProject, status, projectId]);

  // Sync thumbnail once when video becomes ready
  useEffect(() => {
    if (!projectId || !thumbnailUrl || thumbnailSynced.current) return;
    void patchProject({ thumbnail_url: thumbnailUrl }).then((saved) => {
      if (saved) thumbnailSynced.current = true;
    });
  }, [patchProject, thumbnailUrl, projectId]);
}
