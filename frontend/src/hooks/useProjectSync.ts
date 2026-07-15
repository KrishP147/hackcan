"use client";

import { useEffect, useRef } from "react";
import { useUser } from "@auth0/nextjs-auth0/client";

/**
 * When authenticated, debounces frame position and syncs status/thumbnail to Supabase.
 * When not logged in, no API calls — editing works fully without auth.
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
  const { user } = useUser();
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSyncedFrame = useRef<number>(-1);
  const lastSyncedStatus = useRef<string>("");
  const thumbnailSynced = useRef(false);

  // Only sync when authenticated (API requires auth)
  const isAuthenticated = !!user;

  // Idempotently register the project. This also claims a project that was
  // uploaded as a guest immediately before the user chose to sign in.
  useEffect(() => {
    if (!isAuthenticated || !projectId) return;
    fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        name: name || "Untitled Project",
        thumbnail_url: thumbnailUrl || null,
      }),
    }).catch(() => {});
  }, [isAuthenticated, projectId, name, thumbnailUrl]);

  // Debounced frame position sync
  useEffect(() => {
    if (!isAuthenticated || !projectId || !videoLoaded) return;
    if (currentFrame === lastSyncedFrame.current) return;

    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      lastSyncedFrame.current = currentFrame;
      fetch(`/api/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ last_frame: currentFrame }),
      }).catch(() => {});
    }, 5000);

    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [isAuthenticated, currentFrame, projectId, videoLoaded]);

  // Sync status changes to Supabase
  useEffect(() => {
    if (!isAuthenticated || !projectId || !status) return;
    if (status === lastSyncedStatus.current) return;
    lastSyncedStatus.current = status;

    fetch(`/api/projects/${projectId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }).catch(() => {});
  }, [isAuthenticated, status, projectId]);

  // Sync thumbnail once when video becomes ready
  useEffect(() => {
    if (!isAuthenticated || !projectId || !thumbnailUrl || thumbnailSynced.current) return;
    thumbnailSynced.current = true;

    fetch(`/api/projects/${projectId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thumbnail_url: thumbnailUrl }),
    }).catch(() => {});
  }, [isAuthenticated, thumbnailUrl, projectId]);
}
