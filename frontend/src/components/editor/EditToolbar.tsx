"use client";

import { useState, useRef } from "react";
import {
  Palette,
  Maximize2,
  Trash2,
  EyeOff,
  ImagePlus,
  Move,
  Droplet,
  Sun,
  RefreshCw,
  Info,
} from "lucide-react";

export type EditAction =
  | "recolor"
  | "resize"
  | "delete"
  | "blur_region"
  | "move"
  | "color_pop"
  | "glow"
  | "replace"
  | "bg_replace"

interface EditOption {
  id: EditAction;
  icon: React.ElementType;
  label: string;
  needsColor?: boolean;
  needsPrompt?: boolean;
  needsScale?: boolean;
  needsOffset?: boolean;
  category: "object" | "frame";
}

const EDIT_OPTIONS: EditOption[] = [
  { id: "delete", icon: Trash2, label: "Remove", category: "object" },
  { id: "recolor", icon: Palette, label: "Recolor", needsColor: true, category: "object" },
  { id: "resize", icon: Maximize2, label: "Resize", needsScale: true, category: "object" },
  { id: "blur_region", icon: EyeOff, label: "Blur", category: "object" },
  { id: "move", icon: Move, label: "Move", needsOffset: true, category: "object" },
  { id: "color_pop", icon: Droplet, label: "Color Pop", category: "object" },
  { id: "glow", icon: Sun, label: "Glow", category: "object" },
  { id: "replace", icon: RefreshCw, label: "Replace", needsPrompt: true, category: "object" },
  { id: "bg_replace", icon: ImagePlus, label: "Replace BG", needsPrompt: true, category: "frame" },
];

const COLOR_PRESETS = [
  "#F43F5E", "#EF4444", "#F59E0B", "#10B981",
  "#0EA5E9", "#8B5CF6", "#EC4899", "#FFFFFF",
  "#171717", "#6366F1", "#14B8A6", "#F97316",
];

interface EditToolbarProps {
  objectLabel: string;
  active: boolean;
  hasMask: boolean;
  editApplied: boolean;
  isPreviewing: boolean;
  pendingAction?: string | null;
  onApply: (action: EditAction, params: { color?: string; prompt?: string; scale?: number; dx?: number; dy?: number }) => void;
  onUndo: () => void;
  onClose: () => void;
}

export function EditToolbar({ objectLabel, active, hasMask, editApplied, isPreviewing, pendingAction, onApply, onUndo, onClose }: EditToolbarProps) {
  const [selected, setSelected] = useState<EditOption | null>(null);
  const [color, setColor] = useState("#F43F5E");
  const [prompt, setPrompt] = useState("");
  const [scale, setScale] = useState(1.5);
  const [dx, setDx] = useState(0);
  const [dy, setDy] = useState(0);
  const promptInputRef = useRef<HTMLInputElement>(null);

  const objectEdits = EDIT_OPTIONS.filter((o) => o.category === "object");
  const frameEdits = EDIT_OPTIONS.filter((o) => o.category === "frame");
  const selectedRequestPending = Boolean(selected && pendingAction === selected.id);
  const selectedPreviewReady = selectedRequestPending && !isPreviewing;

  const handleApply = () => {
    if (!selected || !active) return;
    if ((selected.category === "object" || selected.id === "bg_replace") && !hasMask) return;
    
    // Read prompt directly from input to avoid stale state
    const currentPrompt = selected.needsPrompt && promptInputRef.current
      ? promptInputRef.current.value
      : prompt;
    
    onApply(selected.id, {
      color: selected.needsColor ? color.replace("#", "") : undefined,
      prompt: selected.needsPrompt ? currentPrompt : undefined,
      scale: selected.needsScale ? scale : undefined,
      dx: selected.needsOffset ? dx : undefined,
      dy: selected.needsOffset ? dy : undefined,
    });
    
  };

  return (
    <div
      className="w-[300px] xl:w-[340px] shrink-0 flex flex-col overflow-y-auto border-l"
      style={{ background: "var(--ed-surface)", borderColor: "var(--ed-border)" }}
      onClick={(e) => e.stopPropagation()}
    >


      {!selected ? (
        <div className="p-5 xl:p-6 flex-1">
          <p
            className="text-xs uppercase tracking-widest font-semibold mb-4 px-1"
            style={{ color: (active && hasMask) ? "var(--ed-icon-dim)" : "var(--ed-disabled)" }}
          >
            Object {!hasMask && active && <span className="normal-case tracking-normal font-normal">(segment first)</span>}
          </p>
          <div className="grid grid-cols-3 gap-2 mb-7">
            {objectEdits.map((opt) => {
              const Icon = opt.icon;
              const enabled = active && hasMask;
              return (
                <button
                  key={opt.id}
                  disabled={!enabled}
                  onClick={() => {
                    if (!opt.needsColor && !opt.needsPrompt && !opt.needsScale && !opt.needsOffset) {
                      onApply(opt.id, {});
                    } else {
                      setSelected(opt);
                      // Reset prompt when selecting a new option
                      setPrompt("");
                    }
                  }}
                  className="flex min-h-20 flex-col items-center justify-center gap-2 py-3 px-1 rounded-xl transition-all"
                  style={{
                    color: enabled ? "var(--ed-icon)" : "var(--ed-disabled)",
                    cursor: enabled ? "pointer" : "not-allowed",
                  }}
                  onMouseEnter={(e) => enabled && (e.currentTarget.style.background = "var(--ed-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <Icon className="w-5 h-5" strokeWidth={1.5} />
                  <span className="text-xs font-medium">{opt.label}</span>
                </button>
              );
            })}
          </div>

          <p
            className="text-xs uppercase tracking-widest font-semibold mb-4 px-1"
            style={{ color: active ? "var(--ed-icon-dim)" : "var(--ed-disabled)" }}
          >
            Whole Frame {!hasMask && active && <span className="normal-case tracking-normal font-normal">(segment foreground first)</span>}
          </p>
          <div className="grid grid-cols-3 gap-2">
            {frameEdits.map((opt) => {
              const Icon = opt.icon;
              const enabled = active && (opt.id !== "bg_replace" || hasMask);
              return (
                <button
                  key={opt.id}
                  disabled={!enabled}
                  onClick={() => {
                    if (!opt.needsColor && !opt.needsPrompt && !opt.needsScale && !opt.needsOffset) {
                      onApply(opt.id, {});
                    } else {
                      setSelected(opt);
                      // Reset prompt when selecting a new option
                      setPrompt("");
                    }
                  }}
                  className="flex min-h-20 flex-col items-center justify-center gap-2 py-3 px-1 rounded-xl transition-all"
                  style={{
                    color: enabled ? "var(--ed-icon)" : "var(--ed-disabled)",
                    cursor: enabled ? "pointer" : "not-allowed",
                  }}
                  onMouseEnter={(e) => enabled && (e.currentTarget.style.background = "var(--ed-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <Icon className="w-5 h-5" strokeWidth={1.5} />
                  <span className="text-xs font-medium">{opt.label}</span>
                </button>
              );
            })}
          </div>

          <div
            className="mt-7 flex gap-3 rounded-2xl border px-4 py-4"
            style={{
              background: "var(--ed-surface-2)",
              borderColor: "var(--ed-border)",
              color: "var(--ed-subtle)",
            }}
          >
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-[var(--accent)]" />
            <p className="text-xs leading-5">
              <span className="font-semibold" style={{ color: "var(--ed-muted)" }}>
                {editApplied ? "Edit saved — keep going." : "You can stack edits."}
              </span>{" "}
              After propagation, choose another tool. Its preview starts from your latest edited result.
            </p>
          </div>

        </div>
      ) : (
        <div className="p-6 space-y-5 flex-1">
          <button
            onClick={() => setSelected(null)}
            className="text-xs transition-colors"
            style={{ color: "var(--ed-subtle)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--ed-muted)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--ed-subtle)")}
          >
            ← Back
          </button>

          <div className="flex items-center gap-2">
            {(() => { const Icon = selected.icon; return <Icon className="w-4 h-4 text-[var(--accent)]" strokeWidth={1.5} />; })()}
            <span className="text-sm font-medium" style={{ color: "var(--ed-text)" }}>
              {selected.label}
            </span>
          </div>

          {selected.needsColor && (
            <div>
              <p className="text-xs mb-2" style={{ color: "var(--ed-subtle)" }}>Color</p>
              <div className="grid grid-cols-6 gap-1.5 mb-2">
                {COLOR_PRESETS.map((c) => (
                  <button
                    key={c}
                    onClick={() => setColor(c)}
                    className={`w-7 h-7 rounded-lg border-2 transition-all ${color === c ? "scale-110" : "hover:scale-105"}`}
                    style={{
                      backgroundColor: c,
                      borderColor: color === c ? "var(--ed-text)" : "transparent",
                    }}
                  />
                ))}
              </div>
              <input
                type="text"
                value={color}
                onChange={(e) => setColor(e.target.value)}
                placeholder="#FF0000"
                className="w-full rounded-xl px-3 py-1.5 text-xs outline-none transition-colors border"
                style={{
                  background: "var(--ed-surface-2)",
                  color: "var(--ed-text)",
                  borderColor: "var(--ed-border)",
                }}
              />
            </div>
          )}

          {selected.needsPrompt && (
            <div>
              <p className="text-xs mb-2" style={{ color: "var(--ed-subtle)" }}>
                {selected.id === "bg_replace" ? "New background…" : "Describe…"}
              </p>
              <input
                ref={promptInputRef}
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={
                  selected.id === "bg_replace" ? "e.g. sunset beach" :
                  "Describe what you want…"
                }
                className="w-full rounded-xl px-3 py-2 text-xs outline-none transition-colors border"
                style={{
                  background: "var(--ed-surface-2)",
                  color: "var(--ed-text)",
                  borderColor: "var(--ed-border)",
                }}
                autoFocus
                onKeyDown={(e) => e.key === "Enter" && handleApply()}
              />
            </div>
          )}

          {selected.needsOffset && (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-2">
                  <p className="text-xs" style={{ color: "var(--ed-subtle)" }}>Horizontal</p>
                  <p className="text-xs font-mono" style={{ color: "var(--ed-muted)" }}>{dx}px</p>
                </div>
                <input
                  type="range" min="-300" max="300" step="5"
                  value={dx}
                  onChange={(e) => setDx(parseInt(e.target.value, 10))}
                  className="w-full accent-[var(--accent)]"
                />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <p className="text-xs" style={{ color: "var(--ed-subtle)" }}>Vertical</p>
                  <p className="text-xs font-mono" style={{ color: "var(--ed-muted)" }}>{dy}px</p>
                </div>
                <input
                  type="range" min="-300" max="300" step="5"
                  value={dy}
                  onChange={(e) => setDy(parseInt(e.target.value, 10))}
                  className="w-full accent-[var(--accent)]"
                />
              </div>
            </div>
          )}

          {selected.needsScale && (
            <div>
              <div className="flex justify-between mb-2">
                <p className="text-xs" style={{ color: "var(--ed-subtle)" }}>Scale</p>
                <p className="text-xs font-mono" style={{ color: "var(--ed-muted)" }}>{scale.toFixed(1)}x</p>
              </div>
              <input
                type="range" min="0.3" max="3.0" step="0.1"
                value={scale}
                onChange={(e) => setScale(parseFloat(e.target.value))}
                className="w-full accent-[var(--accent)]"
              />
            </div>
          )}

          <button
            onClick={handleApply}
            disabled={!active || isPreviewing || selectedPreviewReady || (selected.needsPrompt && !prompt.trim())}
            className="w-full py-2.5 rounded-xl text-white text-sm font-semibold transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: "var(--accent)",
              boxShadow: "0 4px 16px rgba(244,63,94,0.25)",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--accent-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
          >
            {isPreviewing && selectedRequestPending
              ? "Gemini is generating…"
              : selectedPreviewReady
                ? "Preview ready"
                : `Preview ${selected.label}`}
          </button>

          {selectedRequestPending && (
            <div
              className="flex gap-3 rounded-2xl border px-4 py-3"
              style={{
                background: "var(--ed-surface-2)",
                borderColor: "var(--ed-border)",
                color: "var(--ed-subtle)",
              }}
            >
              {isPreviewing ? (
                <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-[var(--accent)]" />
              ) : (
                <span className="mt-0.5 h-4 w-4 shrink-0 text-center text-emerald-500">✓</span>
              )}
              <p className="text-xs leading-5">
                {isPreviewing
                  ? "Prompt received. Gemini is creating one keyframe preview; this usually takes 5–20 seconds."
                  : "Preview ready. Confirm “Apply to all frames” on the video, or cancel and revise your prompt."}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
