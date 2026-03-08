"use client";

import { useState } from "react";
import { Play, ChevronLeft, Sun, Moon, Save, Check, Undo2 } from "lucide-react";
import { useUser } from "@auth0/nextjs-auth0/client";

interface EditorTopBarProps {
  videoName: string;
  onNameChange: (name: string) => void;
  videoLoaded: boolean;
  isDark: boolean;
  onToggleTheme: () => void;
  editApplied?: boolean;
  onUndo?: () => void;
  /** When provided, Save compiles and exports video as MP4 (render + download). */
  onSave?: () => void | Promise<void>;
  isExporting?: boolean;
}

export function EditorTopBar({
  videoName,
  onNameChange,
  videoLoaded,
  isDark,
  onToggleTheme,
  editApplied,
  onUndo,
  onSave,
  isExporting = false,
}: EditorTopBarProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showSignInPrompt, setShowSignInPrompt] = useState(false);
  const { user } = useUser();

  async function handleSave() {
    if (onSave) {
      await onSave();
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      return;
    }
    if (!user) {
      setShowSignInPrompt(true);
      return;
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

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
          href="/dashboard"
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

      {/* Right: Save + Undo + Theme toggle */}
      <div className="flex items-center gap-2">
        {videoLoaded && editApplied && onUndo && (
          <button
            onClick={onUndo}
            className="flex items-center gap-1.5 px-3 h-9 rounded-xl text-sm font-medium transition-all hover:scale-105"
            style={{
              background: "var(--ed-overlay)",
              color: "var(--ed-icon)",
              border: "1px solid var(--ed-border)",
            }}
            title="Undo last change"
          >
            <Undo2 className="w-3.5 h-3.5" />
            Undo
          </button>
        )}
        {videoLoaded && (
          <button
            onClick={() => handleSave()}
            disabled={isExporting}
            className="flex items-center gap-1.5 px-3 h-9 rounded-xl text-sm font-semibold transition-all hover:scale-105 disabled:opacity-70 disabled:pointer-events-none"
            style={{
              background: saved ? "rgba(16,185,129,0.15)" : "rgba(244,63,94,0.15)",
              color: saved ? "#10B981" : "var(--accent)",
              border: `1px solid ${saved ? "rgba(16,185,129,0.3)" : "rgba(244,63,94,0.3)"}`,
            }}
          >
            {saved ? <Check className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
            {isExporting ? "Exporting…" : saved ? "Saved" : "Save"}
          </button>
        )}

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
      </div>

      {/* Sign-in prompt modal */}
      {showSignInPrompt && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowSignInPrompt(false)}
        >
          <div
            className="bg-white rounded-2xl p-8 max-w-sm w-full mx-4 text-center shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="w-14 h-14 rounded-2xl bg-[rgba(244,63,94,0.1)] flex items-center justify-center mx-auto mb-4">
              <Save className="w-6 h-6 text-[#F43F5E]" />
            </div>
            <h2 className="text-xl font-[550] text-[#171717] mb-2">Sign in to save</h2>
            <p className="text-[#6B7280] text-sm mb-6">
              Create a free account to save your projects and pick up where you left off.
            </p>
            <div className="flex flex-col gap-3">
              <a
                href={`/api/auth/login?returnTo=${encodeURIComponent(window.location.pathname)}`}
                className="bg-[#F43F5E] hover:bg-[#E11D48] text-white font-semibold text-sm px-6 py-3 rounded-xl transition-all duration-300 hover:scale-[1.02]"
              >
                Sign in / Create account
              </a>
              <button
                onClick={() => setShowSignInPrompt(false)}
                className="text-sm text-[#6B7280] hover:text-[#171717] transition-colors"
              >
                Continue without saving
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
