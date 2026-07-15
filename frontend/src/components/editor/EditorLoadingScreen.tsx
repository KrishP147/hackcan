"use client";

import { useEffect, useState } from "react";

const CELLS = 8;

/**
 * Full-screen overlay shown between upload and the editor being ready.
 * A strip of frame cells fills in sequence — the product's own metaphor:
 * the backend is literally extracting the clip into frames right now.
 * Fades out over 300ms once `show` flips false, then unmounts.
 */
export function EditorLoadingScreen({ show }: { show: boolean }) {
  const [mounted, setMounted] = useState(show);

  useEffect(() => {
    if (show) {
      setMounted(true);
      return;
    }
    const t = setTimeout(() => setMounted(false), 300);
    return () => clearTimeout(t);
  }, [show]);

  if (!mounted) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-8 transition-opacity duration-300"
      style={{ background: "var(--ed-bg)", opacity: show ? 1 : 0 }}
      role="status"
      aria-live="polite"
    >
      <span
        className="text-sm font-semibold tracking-[0.25em] uppercase"
        style={{ color: "var(--ed-muted)" }}
      >
        FrameShift
      </span>

      <div className="loading-filmstrip" aria-hidden="true">
        {Array.from({ length: CELLS }, (_, i) => (
          <span
            key={i}
            className="loading-filmstrip-cell"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>

      <div className="flex flex-col items-center gap-1.5">
        <p className="text-base font-medium" style={{ color: "var(--ed-muted)" }}>
          Preparing your video
        </p>
        <p className="text-sm" style={{ color: "var(--ed-subtle, #6b7280)" }}>
          Extracting frames&hellip;
        </p>
      </div>
    </div>
  );
}
