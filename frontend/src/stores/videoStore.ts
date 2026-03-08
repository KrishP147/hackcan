"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface VideoProject {
  projectId: string;
  videoName: string;
  uploadedAt: number;
  status?: string;
  frameCount?: number;
}

interface VideoStore {
  projects: VideoProject[];
  currentProjectId: string | null;
  
  // Actions
  addProject: (project: VideoProject) => void;
  setCurrentProject: (projectId: string | null) => void;
  updateProject: (projectId: string, updates: Partial<VideoProject>) => void;
  removeProject: (projectId: string) => void;
  getProject: (projectId: string) => VideoProject | undefined;
}

export const useVideoStore = create<VideoStore>()(
  persist(
    (set, get) => ({
      projects: [],
      currentProjectId: null,

      addProject: (project) =>
        set((state) => {
          // Check if project already exists
          const existingIndex = state.projects.findIndex(
            (p) => p.projectId === project.projectId
          );
          
          if (existingIndex >= 0) {
            // Update existing project
            const updated = [...state.projects];
            updated[existingIndex] = { ...updated[existingIndex], ...project };
            return { projects: updated };
          }
          
          // Add new project
          return {
            projects: [...state.projects, project],
            currentProjectId: project.projectId,
          };
        }),

      setCurrentProject: (projectId) =>
        set({ currentProjectId: projectId }),

      updateProject: (projectId, updates) =>
        set((state) => {
          const index = state.projects.findIndex((p) => p.projectId === projectId);
          if (index < 0) return state;
          
          const updated = [...state.projects];
          updated[index] = { ...updated[index], ...updates };
          return { projects: updated };
        }),

      removeProject: (projectId) =>
        set((state) => ({
          projects: state.projects.filter((p) => p.projectId !== projectId),
          currentProjectId:
            state.currentProjectId === projectId ? null : state.currentProjectId,
        })),

      getProject: (projectId) => {
        return get().projects.find((p) => p.projectId === projectId);
      },
    }),
    {
      name: "video-storage", // localStorage key
    }
  )
);
