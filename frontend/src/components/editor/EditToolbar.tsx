"use client";

import { useState } from "react";
import {
  Palette,
  Maximize2,
  Replace,
  Trash2,
  EyeOff,
  ImageOff,
  ImagePlus,
  Sparkles,
  ArrowUpCircle,
  WandSparkles,
  Droplets,
  X,
} from "lucide-react";

export type EditAction =
  | "recolor"
  | "resize"
  | "replace"
  | "delete"
  | "blur_region"
  | "bg_remove"
  | "bg_replace"
  | "enhance"
  | "upscale"
  | "restore"
  | "gen_recolor";

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
  { id: "replace", icon: Replace, label: "Replace", needsPrompt: true, category: "object" },
  { id: "resize", icon: Maximize2, label: "Resize", needsScale: true, category: "object" },
  { id: "blur_region", icon: EyeOff, label: "Blur", category: "object" },
  { id: "gen_recolor", icon: Droplets, label: "AI Recolor", needsPrompt: true, needsColor: true, category: "object" },
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
  onApply: (action: EditAction, params: { color?: string; prompt?: string; scale?: number }) => void;
  onClose: () => void;
}

export function EditToolbar({ objectLabel, active, onApply, onClose }: EditToolbarProps) {
  const [selected, setSelected] = useState<EditOption | null>(null);
  const [color, setColor] = useState("#F43F5E");
  const [prompt, setPrompt] = useState("");
  const [scale, setScale] = useState(1.5);

  const objectEdits = EDIT_OPTIONS.filter((o) => o.category === "object");
  const frameEdits = EDIT_OPTIONS.filter((o) => o.category === "frame");

  const handleApply = () => {
    if (!selected || !active) return;
    onApply(selected.id, {
      color: selected.needsColor ? color.replace("#", "") : undefined,
      prompt: selected.needsPrompt ? prompt : undefined,
      scale: selected.needsScale ? scale : undefined,
    });
  };

  return (
    <div
      className="w-[220px] shrink-0 flex flex-col overflow-y-auto border-l"
      style={{ background: "var(--ed-surface)", borderColor: "var(--ed-border)" }}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b"
        style={{ borderColor: "var(--ed-surface-2)" }}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-1.5 h-1.5 rounded-full transition-colors"
            style={{ background: active ? "var(--accent)" : "var(--ed-icon-dim)" }}
          />
          <span
            className="text-sm font-medium transition-colors"
            style={{ color: active ? "var(--ed-text)" : "var(--ed-subtle)" }}
          >
            {active ? objectLabel : "selection"}
          </span>
        </div>
        {/* no close button */}
      </div>

      {!selected ? (
        <div className="p-3 flex-1">
          <p
            className="text-[10px] uppercase tracking-widest font-semibold mb-2 px-1"
            style={{ color: active ? "var(--ed-icon-dim)" : "var(--ed-disabled)" }}
          >
            Object
          </p>
          <div className="grid grid-cols-3 gap-1 mb-4">
            {objectEdits.map((opt) => {
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
                {selected.id === "replace" ? "Replace with…" : selected.id === "bg_replace" ? "New background…" : "Describe…"}
              </p>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={
                  selected.id === "replace" ? "e.g. a red sports car" :
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
