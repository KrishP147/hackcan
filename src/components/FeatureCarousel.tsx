"use client";

import { useState, useEffect, useCallback } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useInView } from "@/hooks/useInView";

/* ─── Feature Data ─── */
const FEATURES = [
  {
    label: "DETECT",
    title: "AI Object Detection",
    description:
      "Click on any object in the frame and the AI instantly selects it — no manual masking, no rotoscoping.",
    color: "#F43F5E",
    previewBg: "#0f1623",
  },
  {
    label: "REMOVE",
    title: "Background Removal",
    description:
      "Remove or replace any background instantly. The AI fills what's behind the subject in real time.",
    color: "#0EA5E9",
    previewBg: "#0a1520",
  },
  {
    label: "RECOLOR",
    title: "Color Grading",
    description:
      "Change the color of any detected object in seconds — pick a hue and watch it transform.",
    color: "#F59E0B",
    previewBg: "#141008",
  },
  {
    label: "RESIZE",
    title: "Object Resize",
    description:
      "Scale any object up or down inside your video. Set your timeline range and AI keeps it seamless frame-to-frame.",
    color: "#10B981",
    previewBg: "#091812",
  },
];

const AUTO_MS = 5000;

/* ─── Animated Previews ─── */
function DetectPreview({ color }: { color: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
      {/* Animated scan line */}
      <div
        className="absolute left-12 right-12"
        style={{
          height: "2px",
          background: `linear-gradient(90deg, transparent, ${color}90, transparent)`,
          animation: "preview-scan 3s ease-in-out infinite",
        }}
      />

      {/* Person silhouette */}
      <div className="relative flex flex-col items-center gap-1">
        {/* Head */}
        <div
          className="w-12 h-12 rounded-full"
          style={{
            backgroundColor: `${color}12`,
            border: `2px solid ${color}35`,
          }}
        />
        {/* Body */}
        <div
          className="w-20 h-24 rounded-t-3xl"
          style={{
            backgroundColor: `${color}08`,
            border: `2px solid ${color}25`,
          }}
        />

        {/* Selection box */}
        <div
          className="absolute -inset-6 rounded-2xl"
          style={{
            border: `2px solid ${color}`,
            animation: "preview-pulse 2s ease-in-out infinite",
          }}
        />

        {/* Corner brackets */}
        <div className="absolute -top-0.5 -left-0.5 w-4 h-4" style={{ borderTop: `2px solid ${color}`, borderLeft: `2px solid ${color}`, borderRadius: "4px 0 0 0" }} />
        <div className="absolute -top-0.5 -right-0.5 w-4 h-4" style={{ borderTop: `2px solid ${color}`, borderRight: `2px solid ${color}`, borderRadius: "0 4px 0 0" }} />
        <div className="absolute -bottom-0.5 -left-0.5 w-4 h-4" style={{ borderBottom: `2px solid ${color}`, borderLeft: `2px solid ${color}`, borderRadius: "0 0 0 4px" }} />
        <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4" style={{ borderBottom: `2px solid ${color}`, borderRight: `2px solid ${color}`, borderRadius: "0 0 4px 0" }} />

        {/* Detection badge */}
        <div
          className="absolute -top-12 left-1/2 -translate-x-1/2 px-3 py-1 rounded-lg text-white text-xs font-bold tracking-widest whitespace-nowrap"
          style={{ backgroundColor: color }}
        >
          PERSON · 96%
        </div>
      </div>
    </div>
  );
}

function RemovePreview({ color }: { color: string }) {
  return (
    <div className="absolute inset-0 overflow-hidden">
      {/* Before: left half */}
      <div
        className="absolute inset-0 flex items-center justify-center"
        style={{ clipPath: "inset(0 50% 0 0)" }}
      >
        <div className="w-16 h-36 rounded-t-full" style={{ backgroundColor: `${color}30`, border: `2px solid ${color}50` }} />
      </div>

      {/* After: right half (transparent checkerboard) */}
      <div
        className="absolute inset-0"
        style={{
          clipPath: "inset(0 0 0 50%)",
          backgroundImage: `
            linear-gradient(45deg, #1a2332 25%, transparent 25%),
            linear-gradient(-45deg, #1a2332 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #1a2332 75%),
            linear-gradient(-45deg, transparent 75%, #1a2332 75%)
          `,
          backgroundSize: "18px 18px",
          backgroundPosition: "0 0, 0 9px, 9px -9px, -9px 0",
          backgroundColor: "#0d1117",
        }}
      />

      {/* Center divider */}
      <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-0.5" style={{ backgroundColor: color }} />

      {/* Labels */}
      <span className="absolute bottom-4 left-5 text-white/30 text-[10px] font-semibold uppercase tracking-widest">
        Before
      </span>
      <span
        className="absolute bottom-4 right-5 text-[10px] font-semibold uppercase tracking-widest"
        style={{ color: `${color}70` }}
      >
        After
      </span>
    </div>
  );
}

function RecolorPreview({ color }: { color: string }) {
  const swatches = ["#F43F5E", "#0EA5E9", "#F59E0B", "#10B981", "#F97316"];

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-8">
      {/* Hue-rotating sphere */}
      <div
        className="w-32 h-32 rounded-full"
        style={{
          backgroundColor: color,
          animation: "preview-hue 8s linear infinite",
          boxShadow: `0 0 48px ${color}50`,
        }}
      />

      {/* Color swatches */}
      <div className="flex items-center gap-3">
        {swatches.map((c) => (
          <div
            key={c}
            className="rounded-full"
            style={{
              backgroundColor: c,
              width: c === color ? "28px" : "18px",
              height: c === color ? "28px" : "18px",
              boxShadow: c === color ? `0 0 12px ${c}80` : "none",
              transition: "all 0.4s cubic-bezier(0.22, 1, 0.36, 1)",
            }}
          />
        ))}
      </div>
    </div>
  );
}

function ResizePreview({ color }: { color: string }) {
  const [phase, setPhase] = useState(0); // 0=normal, 1=enlarged

  useEffect(() => {
    const t = setInterval(() => setPhase((p) => (p + 1) % 2), 2000);
    return () => clearInterval(t);
  }, []);

  const objectSize = phase === 0 ? 56 : 96;

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-0">
      {/* Video frame mock */}
      <div
        className="relative flex items-center justify-center overflow-hidden"
        style={{
          width: "260px",
          height: "146px",
          border: `1px solid ${color}20`,
          borderRadius: "8px",
          backgroundColor: "rgba(0,0,0,0.3)",
        }}
      >
        {/* Object being resized */}
        <div
          className="relative flex items-center justify-center rounded-xl"
          style={{
            width: objectSize,
            height: objectSize,
            backgroundColor: `${color}20`,
            border: `2px solid ${color}`,
            transition: "all 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        >
          {/* Corner resize handles */}
          {[
            { top: "-4px", left: "-4px" },
            { top: "-4px", right: "-4px" },
            { bottom: "-4px", left: "-4px" },
            { bottom: "-4px", right: "-4px" },
          ].map((pos, i) => (
            <div
              key={i}
              className="absolute w-2.5 h-2.5 rounded-sm"
              style={{ backgroundColor: color, ...pos }}
            />
          ))}

          {/* Size label */}
          <span
            className="font-mono font-bold"
            style={{ fontSize: "9px", color: `${color}cc` }}
          >
            {phase === 0 ? "1×" : "1.7×"}
          </span>
        </div>

        {/* Selection label */}
        <div
          className="absolute top-2 left-2 px-1.5 py-0.5 rounded text-white font-semibold"
          style={{ backgroundColor: color, fontSize: "8px" }}
        >
          PERSON
        </div>
      </div>

      {/* Timeline range indicator */}
      <div
        className="mt-3 flex flex-col gap-1"
        style={{ width: "260px" }}
      >
        <div className="flex justify-between" style={{ fontSize: "8px", color: `${color}60` }}>
          <span>0:00</span>
          <span className="font-semibold" style={{ color: `${color}cc` }}>
            Range: 0:04 – 0:12
          </span>
          <span>0:30</span>
        </div>
        {/* Full timeline bar */}
        <div className="h-1.5 rounded-full w-full" style={{ backgroundColor: `${color}15` }}>
          {/* Selected range */}
          <div
            className="h-full rounded-full"
            style={{
              marginLeft: "13.3%",
              width: "26.7%",
              backgroundColor: color,
            }}
          />
        </div>
        <p className="text-center" style={{ fontSize: "8px", color: `${color}50` }}>
          Resize persists throughout selected range
        </p>
      </div>
    </div>
  );
}

function FeaturePreview({
  label,
  color,
}: {
  label: string;
  color: string;
}) {
  if (label === "DETECT") return <DetectPreview color={color} />;
  if (label === "REMOVE") return <RemovePreview color={color} />;
  if (label === "RECOLOR") return <RecolorPreview color={color} />;
  if (label === "RESIZE") return <ResizePreview color={color} />;
  return null;
}

/* ─── Main Carousel ─── */
export function FeatureCarousel() {
  const [active, setActive] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const { ref: sectionRef, inView } = useInView();

  const next = useCallback(
    () => setActive((p) => (p + 1) % FEATURES.length),
    []
  );
  const prev = useCallback(
    () => setActive((p) => (p - 1 + FEATURES.length) % FEATURES.length),
    []
  );

  useEffect(() => {
    if (isPaused || !inView) return;
    const t = setInterval(next, AUTO_MS);
    return () => clearInterval(t);
  }, [isPaused, inView, next]);

  const feature = FEATURES[active];
  const prevIdx = (active - 1 + FEATURES.length) % FEATURES.length;
  const nextIdx = (active + 1) % FEATURES.length;

  return (
    <section
      ref={sectionRef}
      className="py-24 px-6 md:px-12 lg:px-24 bg-[var(--bg-subtle)]"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      {/* Heading */}
      <div className="text-center mb-14">
        <h2
          className={`font-extrabold tracking-tight leading-none text-[clamp(2rem,8vw,4.5rem)] ${
            inView ? "animate-logo-intro" : "opacity-0"
          }`}
        >
          AI-powered tools for{" "}
          <span className={`keyword-highlight animate-gradient-text`}>
            every
          </span>{" "}
          edit.
        </h2>
      </div>

      {/* Carousel with side peeks */}
      <div
        className={`max-w-6xl mx-auto ${inView ? "animate-fade-up" : "opacity-0"}`}
        style={{ animationDelay: inView ? "0.2s" : undefined }}
      >
        {/* Three-column: peek | active | peek */}
        <div className="flex items-stretch gap-3">

          {/* Left peek — previous card */}
          <button
            onClick={prev}
            aria-label="Previous feature"
            className="hidden md:flex flex-col justify-end flex-shrink-0 w-28 lg:w-36 rounded-xl overflow-hidden cursor-pointer transition-all duration-300 hover:opacity-80 hover:scale-[1.01]"
            style={{
              opacity: 0.45,
              backgroundColor: FEATURES[prevIdx].previewBg,
              background: `radial-gradient(ellipse at 50% 40%, ${FEATURES[prevIdx].color}18, ${FEATURES[prevIdx].previewBg} 70%)`,
            }}
          >
            {/* Fade edge */}
            <div
              className="absolute inset-y-0 right-0 w-12 pointer-events-none"
              style={{ background: `linear-gradient(to right, transparent, var(--bg-subtle))` }}
            />
            <div className="p-3 relative z-10">
              <span
                className="text-[9px] font-bold uppercase tracking-widest"
                style={{ color: FEATURES[prevIdx].color }}
              >
                {FEATURES[prevIdx].label}
              </span>
            </div>
            <div className="h-0.5 w-full" style={{ backgroundColor: FEATURES[prevIdx].color }} />
          </button>

          {/* Active card */}
          <div className="flex-1 relative aspect-video rounded-2xl overflow-hidden">
            {FEATURES.map((f, i) => (
              <div
                key={f.label}
                className="absolute inset-0 transition-all duration-500"
                style={{
                  transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
                  opacity: i === active ? 1 : 0,
                  transform: i === active ? "scale(1)" : "scale(1.03)",
                  backgroundColor: f.previewBg,
                  pointerEvents: i === active ? "auto" : "none",
                }}
              >
                <FeaturePreview label={f.label} color={f.color} />

                {/* Bottom accent line */}
                <div
                  className="absolute bottom-0 left-0 right-0 h-1"
                  style={{ backgroundColor: f.color }}
                />
              </div>
            ))}

            {/* Edit video button */}
            <button
              className="absolute bottom-4 right-4 z-10 flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold cursor-pointer transition-all duration-300 hover:brightness-110"
              style={{
                backgroundColor: "rgba(0,0,0,0.5)",
                border: `1px solid ${feature.color}55`,
                color: feature.color,
                backdropFilter: "blur(10px)",
                WebkitBackdropFilter: "blur(10px)",
              }}
            >
              Edit video
            </button>
          </div>

          {/* Right peek — next card */}
          <button
            onClick={next}
            aria-label="Next feature"
            className="hidden md:flex flex-col justify-end flex-shrink-0 w-28 lg:w-36 rounded-xl overflow-hidden cursor-pointer transition-all duration-300 hover:opacity-80 hover:scale-[1.01]"
            style={{
              opacity: 0.45,
              backgroundColor: FEATURES[nextIdx].previewBg,
              background: `radial-gradient(ellipse at 50% 40%, ${FEATURES[nextIdx].color}18, ${FEATURES[nextIdx].previewBg} 70%)`,
            }}
          >
            {/* Fade edge */}
            <div
              className="absolute inset-y-0 left-0 w-12 pointer-events-none"
              style={{ background: `linear-gradient(to left, transparent, var(--bg-subtle))` }}
            />
            <div className="p-3 relative z-10">
              <span
                className="text-[9px] font-bold uppercase tracking-widest"
                style={{ color: FEATURES[nextIdx].color }}
              >
                {FEATURES[nextIdx].label}
              </span>
            </div>
            <div className="h-0.5 w-full" style={{ backgroundColor: FEATURES[nextIdx].color }} />
          </button>

        </div>

        {/* Centered navigation row */}
        <div className="flex flex-col items-center gap-6 mt-8">
          {/* Tab buttons + arrows */}
          <div className="flex items-center gap-3">
            {/* Prev arrow */}
            <button
              onClick={prev}
              className="w-10 h-10 rounded-full border border-[var(--border)] flex items-center justify-center hover:border-[var(--accent)] hover:text-[var(--accent)] cursor-pointer"
              aria-label="Previous feature"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            {/* Feature tab buttons */}
            {FEATURES.map((f, i) => (
              <button
                key={f.label}
                onClick={() => setActive(i)}
                className="px-4 py-2 rounded-lg text-sm font-semibold uppercase tracking-wider cursor-pointer"
                style={{
                  backgroundColor:
                    i === active ? f.color : "transparent",
                  color: i === active ? "#fff" : "var(--fg-muted)",
                  border:
                    i === active
                      ? `1px solid ${f.color}`
                      : "1px solid var(--border)",
                  transition: "all 300ms cubic-bezier(0.22, 1, 0.36, 1)",
                }}
              >
                {f.label}
              </button>
            ))}

            {/* Next arrow */}
            <button
              onClick={next}
              className="w-10 h-10 rounded-full border border-[var(--border)] flex items-center justify-center hover:border-[var(--accent)] hover:text-[var(--accent)] cursor-pointer"
              aria-label="Next feature"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Progress dots */}
          <div className="flex gap-2 items-center">
            {FEATURES.map((_, i) => (
              <button
                key={i}
                onClick={() => setActive(i)}
                className="h-1.5 rounded-full cursor-pointer"
                style={{
                  width: i === active ? "32px" : "8px",
                  backgroundColor:
                    i === active ? feature.color : "var(--border)",
                  transition: "all 400ms cubic-bezier(0.22, 1, 0.36, 1)",
                }}
                aria-label={`Go to feature ${i + 1}`}
              />
            ))}
          </div>

          {/* Feature title + description */}
          <div className="text-center max-w-lg relative min-h-[90px]">
            {FEATURES.map((f, i) => (
              <div
                key={f.label}
                className="transition-all duration-500"
                style={{
                  transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
                  opacity: i === active ? 1 : 0,
                  transform:
                    i === active ? "translateY(0)" : "translateY(8px)",
                  position: i === active ? "relative" : "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  pointerEvents: i === active ? "auto" : "none",
                }}
              >
                <h3
                  className="text-xl font-[550] mb-2"
                  style={{ color: "var(--fg)" }}
                >
                  {f.title}
                </h3>
                <p className="text-[var(--fg-muted)] leading-relaxed">
                  {f.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
