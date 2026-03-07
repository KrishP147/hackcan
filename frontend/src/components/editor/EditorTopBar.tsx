"use client";

import { useState } from "react";
import { Play, ChevronLeft, Sun, Moon } from "lucide-react";

interface EditorTopBarProps {
  videoName: string;
  onNameChange: (name: string) => void;
  videoLoaded: boolean;
  isDark: boolean;
  onToggleTheme: () => void;
}

export function EditorTopBar({
  videoName,
  onNameChange,
  videoLoaded,
  isDark,
  onToggleTheme,
}: EditorTopBarProps) {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <header
      className="h-14 flex items-center justify-between px-4 border-b shrink-0"
      style={{
        background: "var(--ed-surface)",
        borderColor: "var(--ed-border)",
      }}
    >
      {/* Left: Back + name */}
      <div className="flex items-center gap-3">
        <a
          href="/"
          className="flex items-center gap-2 transition-opacity hover:opacity-70"
          style={{ color: "var(--ed-icon)" }}
        >
          <ChevronLeft className="w-4 h-4" />
          <div
            className="w-7 h-7 rounded-full flex items-center justify-center"
            style={{ background: "rgba(244,63,94,0.15)" }}
          >
            <Play className="w-3 h-3 text-[var(--accent)] ml-0.5" fill="currentColor" />
          </div>
        </a>

        {videoLoaded ? (
          isEditing ? (
            <input
              type="text"
              value={videoName}
              onChange={(e) => onNameChange(e.target.value)}
              onBlur={() => setIsEditing(false)}
              onKeyDown={(e) => e.key === "Enter" && setIsEditing(false)}
              autoFocus
              className="bg-transparent text-sm font-medium border-b border-[var(--accent)] outline-none px-1 py-0.5"
              style={{ color: "var(--ed-text)" }}
            />
          ) : (
            <button
              onClick={() => setIsEditing(true)}
              className="text-sm font-medium hover:text-[var(--accent)] transition-colors px-1 py-0.5"
              style={{ color: "var(--ed-text)" }}
            >
              {videoName}
            </button>
          )
        ) : (
          <span className="text-sm" style={{ color: "var(--ed-subtle)" }}>
            FrameShift Editor
          </span>
        )}
      </div>

      {/* Right: Theme toggle */}
      <button
        onClick={onToggleTheme}
        className="w-9 h-9 flex items-center justify-center rounded-xl transition-all hover:scale-105"
        style={{
          background: "var(--ed-overlay)",
          color: "var(--ed-icon)",
          border: "1px solid var(--ed-border)",
        }}
        title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      >
        {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
      </button>
    </header>
  );
}
