"use client";

import { useState, useRef } from "react";
import {
  Palette,
  Maximize2,
  Trash2,
  EyeOff,
  ImageOff,
  ImagePlus,
  Sparkles,
  ArrowUpCircle,
  WandSparkles,
  Film,
  Undo2,
  Loader2,
} from "lucide-react";

export type EditAction =
  | "recolor"
  | "resize"
  | "delete"
  | "blur_region"
  | "bg_remove"
  | "bg_replace"
  | "enhance"
  | "upscale"
  | "restore"

interface EditOption {
  id: EditAction;
  icon: React.ElementType;
  label: string;
  needsColor?: boolean;
  needsPrompt?: boolean;
  needsScale?: boolean;
  category: "object" | "frame";
}

const EDIT_OPTIONS: EditOption[] = [
  { id: "delete", icon: Trash2, label: "Remove", category: "object" },
  { id: "recolor", icon: Palette, label: "Recolor", needsColor: true, category: "object" },
  { id: "resize", icon: Maximize2, label: "Resize", needsScale: true, category: "object" },
  { id: "blur_region", icon: EyeOff, label: "Blur", category: "object" },
  { id: "bg_remove", icon: ImageOff, label: "Remove BG", category: "frame" },
  { id: "bg_replace", icon: ImagePlus, label: "Replace BG", needsPrompt: true, category: "frame" },
  { id: "enhance", icon: Sparkles, label: "Enhance", category: "frame" },
  { id: "upscale", icon: ArrowUpCircle, label: "Upscale", category: "frame" },
  { id: "restore", icon: WandSparkles, label: "Restore", category: "frame" },
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
  isRefining?: boolean;
  onApply: (action: EditAction, params: { color?: string; prompt?: string; scale?: number }) => void;
  onRefine: () => void;
  onPropagate: (prompt: string) => void;
  onUndo: () => void;
  onClose: () => void;
}

export function EditToolbar({ objectLabel, active, hasMask, editApplied, isRefining = false, onApply, onRefine, onPropagate, onUndo, onClose }: EditToolbarProps) {
  const [selected, setSelected] = useState<EditOption | null>(null);
  const [color, setColor] = useState("#F43F5E");
  const [prompt, setPrompt] = useState("");
  const [scale, setScale] = useState(1.5);
  const promptInputRef = useRef<HTMLInputElement>(null);

  const objectEdits = EDIT_OPTIONS.filter((o) => o.category === "object");
  const frameEdits = EDIT_OPTIONS.filter((o) => o.category === "frame");

  const handleApply = () => {
    if (!selected || !active) return;
    if (selected.category === "object" && !hasMask) return;
    
    // Read prompt directly from input to avoid stale state
    const currentPrompt = selected.needsPrompt && promptInputRef.current
      ? promptInputRef.current.value
      : prompt;
    
    onApply(selected.id, {
      color: selected.needsColor ? color.replace("#", "") : undefined,
      prompt: selected.needsPrompt ? currentPrompt : undefined,
      scale: selected.needsScale ? scale : undefined,
    });
    
    // Reset selection after applying
    setSelected(null);
    setPrompt("");
  };

  return (
    <div
      className="w-[220px] shrink-0 flex flex-col overflow-y-auto border-l"
      style={{ background: "var(--ed-surface)", borderColor: "var(--ed-border)" }}
      onClick={(e) => e.stopPropagation()}
    >


      {!selected ? (
        <div className="p-3 flex-1">
          <p
            className="text-[10px] uppercase tracking-widest font-semibold mb-2 px-1"
            style={{ color: (active && hasMask) ? "var(--ed-icon-dim)" : "var(--ed-disabled)" }}
          >
            Object {!hasMask && active && <span className="normal-case tracking-normal font-normal">(segment first)</span>}
          </p>
          <div className="grid grid-cols-3 gap-1 mb-4">
            {objectEdits.map((opt) => {
              const Icon = opt.icon;
              const enabled = active && hasMask;
              return (
                <button
                  key={opt.id}
                  disabled={!enabled}
                  onClick={() => {
                    if (!opt.needsColor && !opt.needsPrompt && !opt.needsScale) {
                      onApply(opt.id, {});
                    } else {
                      setSelected(opt);
                      // Reset prompt when selecting a new option
                      setPrompt("");
                    }
                  }}
                  className="flex flex-col items-center gap-1.5 py-2.5 px-1 rounded-xl transition-all"
                  style={{
                    color: enabled ? "var(--ed-icon)" : "var(--ed-disabled)",
                    cursor: enabled ? "pointer" : "not-allowed",
                  }}
                  onMouseEnter={(e) => enabled && (e.currentTarget.style.background = "var(--ed-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <Icon className="w-4 h-4" strokeWidth={1.5} />
                  <span className="text-[10px] font-medium">{opt.label}</span>
                </button>
              );
            })}
          </div>

          <p
            className="text-[10px] uppercase tracking-widest font-semibold mb-2 px-1"
            style={{ color: active ? "var(--ed-icon-dim)" : "var(--ed-disabled)" }}
          >
            Whole Frame
          </p>
          <div className="grid grid-cols-3 gap-1">
            {frameEdits.map((opt) => {
              const Icon = opt.icon;
              return (
                <button
                  key={opt.id}
                  disabled={!active}
                  onClick={() => {
                    if (!opt.needsColor && !opt.needsPrompt && !opt.needsScale) {
                      onApply(opt.id, {});
                    } else {
                      setSelected(opt);
                      // Reset prompt when selecting a new option
                      setPrompt("");
                    }
                  }}
                  className="flex flex-col items-center gap-1.5 py-2.5 px-1 rounded-xl transition-all"
                  style={{
                    color: active ? "var(--ed-icon)" : "var(--ed-disabled)",
                    cursor: active ? "pointer" : "not-allowed",
                  }}
                  onMouseEnter={(e) => active && (e.currentTarget.style.background = "var(--ed-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <Icon className="w-4 h-4" strokeWidth={1.5} />
                  <span className="text-[10px] font-medium">{opt.label}</span>
                </button>
              );
            })}
          </div>

          <>
            <div className="my-3 border-t" style={{ borderColor: "var(--ed-border)" }} />
            <p
              className="text-[10px] uppercase tracking-widest font-semibold mb-2 px-1"
              style={{ color: "var(--ed-icon-dim)" }}
            >
              AI Enhance
            </p>
            <button
              onClick={onRefine}
              disabled={isRefining}
              className="flex items-center gap-2 w-full py-2.5 px-3 rounded-xl text-xs font-semibold transition-all border mb-2 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: "var(--accent)",
                color: "#fff",
                borderColor: "transparent",
                boxShadow: "0 4px 16px rgba(244,63,94,0.25)",
              }}
              onMouseEnter={(e) => !isRefining && (e.currentTarget.style.opacity = "0.9")}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
            >
              {isRefining ? <Loader2 className="w-4 h-4 animate-spin" strokeWidth={1.5} /> : null}
              {isRefining ? "Making Realistic..." : "Make Realistic"}
            </button>

          </>
          <button
            onClick={() => {
              const desc = prompt || "Apply the same visual edit consistently";
              onPropagate(desc);
            }}
            className="flex items-center gap-2 w-full py-2.5 px-3 rounded-xl text-xs font-semibold transition-all border mt-2"
            style={{
              background: "var(--accent)",
              color: "#fff",
              borderColor: "transparent",
              boxShadow: "0 4px 16px rgba(244,63,94,0.25)",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.9")}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
          >
            Propagate to All Frames
          </button>
        </div>
      ) : (
        <div className="p-4 space-y-4 flex-1">
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
            disabled={selected.needsPrompt && !prompt.trim()}
            className="w-full py-2.5 rounded-xl text-white text-sm font-semibold transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: "var(--accent)",
              boxShadow: "0 4px 16px rgba(244,63,94,0.25)",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--accent-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--accent)")}
          >
            Apply {selected.label}
          </button>
        </div>
      )}
    </div>
  );
}
