"use client";

interface FeatureCardProps {
  label: string;
  title: string;
  description: string;
  color: string;
  bg: string;
}

function CardVisual({ label, color }: { label: string; color: string }) {
  if (label === "DETECT") {
    return (
      <div className="relative flex items-center justify-center" style={{ height: "160px" }}>
        <div className="relative flex flex-col items-center gap-1.5">
          {/* Head */}
          <div className="w-8 h-8 rounded-full" style={{ backgroundColor: `${color}25`, border: `1.5px solid ${color}40` }} />
          {/* Body */}
          <div className="w-14 h-16 rounded-t-3xl" style={{ backgroundColor: `${color}15`, border: `1.5px solid ${color}30` }} />
          {/* Pulsing selection box */}
          <div
            className="absolute -inset-5 rounded-xl"
            style={{ border: `2px solid ${color}`, animation: "preview-pulse 2s ease-in-out infinite" }}
          />
          {/* Corner brackets */}
          <div className="absolute -top-px -left-px w-3 h-3" style={{ borderTop: `2px solid ${color}`, borderLeft: `2px solid ${color}`, borderRadius: "3px 0 0 0" }} />
          <div className="absolute -top-px -right-px w-3 h-3" style={{ borderTop: `2px solid ${color}`, borderRight: `2px solid ${color}`, borderRadius: "0 3px 0 0" }} />
          <div className="absolute -bottom-px -left-px w-3 h-3" style={{ borderBottom: `2px solid ${color}`, borderLeft: `2px solid ${color}`, borderRadius: "0 0 0 3px" }} />
          <div className="absolute -bottom-px -right-px w-3 h-3" style={{ borderBottom: `2px solid ${color}`, borderRight: `2px solid ${color}`, borderRadius: "0 0 3px 0" }} />
          {/* Detection badge */}
          <div
            className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded text-white font-bold whitespace-nowrap"
            style={{ backgroundColor: color, fontSize: "9px" }}
          >
            PERSON · 96%
          </div>
        </div>
      </div>
    );
  }

  if (label === "REMOVE") {
    return (
      <div className="flex items-center justify-center" style={{ height: "160px" }}>
        <div className="relative overflow-hidden rounded-xl" style={{ width: "160px", height: "90px" }}>
          {/* Checkerboard (after state) fills whole card */}
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `
                linear-gradient(45deg, #1f2937 25%, transparent 25%),
                linear-gradient(-45deg, #1f2937 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #1f2937 75%),
                linear-gradient(-45deg, transparent 75%, #1f2937 75%)
              `,
              backgroundSize: "14px 14px",
              backgroundPosition: "0 0, 0 7px, 7px -7px, -7px 0",
              backgroundColor: "#0a0f15",
            }}
          />
          {/* Before: left half overlay with person */}
          <div
            className="absolute left-0 top-0 bottom-0 flex items-center justify-center"
            style={{ width: "50%", backgroundColor: `${color}10` }}
          >
            <div style={{ width: 32, height: 60, borderRadius: "16px 16px 0 0", backgroundColor: `${color}35`, border: `1.5px solid ${color}50` }} />
          </div>
          {/* Center divider */}
          <div className="absolute top-0 bottom-0 left-1/2 w-px" style={{ backgroundColor: color }} />
          {/* Labels */}
          <span className="absolute bottom-2 left-2 text-white/30 font-semibold uppercase tracking-widest" style={{ fontSize: "8px" }}>Before</span>
          <span className="absolute bottom-2 right-2 font-semibold uppercase tracking-widest" style={{ fontSize: "8px", color: `${color}90` }}>After</span>
        </div>
      </div>
    );
  }

  if (label === "RECOLOR") {
    return (
      <div className="flex flex-col items-center justify-center gap-4" style={{ height: "160px" }}>
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: "50%",
            backgroundColor: color,
            boxShadow: `0 0 36px ${color}60`,
            animation: "preview-hue 6s linear infinite",
          }}
        />
        <div className="flex gap-2 items-center">
          {(["#F43F5E", "#0EA5E9", "#F59E0B", "#10B981", "#F97316"] as const).map((c) => (
            <div
              key={c}
              style={{
                width: c === color ? 18 : 12,
                height: c === color ? 18 : 12,
                borderRadius: "50%",
                backgroundColor: c,
                boxShadow: c === color ? `0 0 8px ${c}80` : "none",
                transition: "all 0.3s",
                flexShrink: 0,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (label === "RESIZE") {
    return (
      <div className="flex items-center justify-center" style={{ height: "160px" }}>
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: 12,
            backgroundColor: `${color}20`,
            border: `2px solid ${color}`,
            animation: "card-resize 2.5s cubic-bezier(0.22, 1, 0.36, 1) infinite",
            position: "relative",
          }}
        >
          {/* Corner handles */}
          <div style={{ position: "absolute", width: 8, height: 8, borderRadius: 2, backgroundColor: color, top: -4, left: -4 }} />
          <div style={{ position: "absolute", width: 8, height: 8, borderRadius: 2, backgroundColor: color, top: -4, right: -4 }} />
          <div style={{ position: "absolute", width: 8, height: 8, borderRadius: 2, backgroundColor: color, bottom: -4, left: -4 }} />
          <div style={{ position: "absolute", width: 8, height: 8, borderRadius: 2, backgroundColor: color, bottom: -4, right: -4 }} />
        </div>
      </div>
    );
  }

  return null;
}

export function FeatureCard({ label, title, description, color, bg }: FeatureCardProps) {
  return (
    <div
      className="relative rounded-2xl overflow-hidden transition-all duration-500 hover:-translate-y-1.5 hover:shadow-2xl"
      style={{
        background: `radial-gradient(ellipse at 50% -5%, ${color}22 0%, ${bg} 60%)`,
        minHeight: "300px",
      }}
    >
      {/* Label badge */}
      <div className="absolute top-5 left-5 z-10">
        <span
          className="px-2.5 py-1 rounded-lg text-xs font-bold uppercase tracking-widest"
          style={{ backgroundColor: `${color}22`, color }}
        >
          {label}
        </span>
      </div>

      {/* Animated preview */}
      <div className="pt-12 px-6">
        <CardVisual label={label} color={color} />
      </div>

      {/* Text */}
      <div className="px-6 pb-6">
        <h3 className="text-xl font-[550] text-white mb-1.5">{title}</h3>
        <p className="text-sm leading-relaxed" style={{ color: "rgba(255,255,255,0.45)" }}>
          {description}
        </p>
      </div>
    </div>
  );
}
