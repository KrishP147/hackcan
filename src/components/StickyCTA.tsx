"use client";

import { useEffect, useState } from "react";
import { Upload } from "lucide-react";

export function StickyCTA() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const dropZone = document.querySelector("[data-dropzone]");
    if (!dropZone) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setVisible(!entry.isIntersecting);
      },
      { threshold: 0 }
    );

    observer.observe(dropZone);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      className={`fixed bottom-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-lg border-t border-gray-100 py-3 px-6 transition-all duration-400 ${
        visible
          ? "opacity-100 translate-y-0"
          : "opacity-0 translate-y-5 pointer-events-none"
      }`}
      style={{ transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)" }}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-center gap-3">
        <button className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-[var(--fg)] text-[var(--fg)] text-sm font-semibold transition-all duration-300 hover:bg-[var(--bg-subtle)] hover:border-[var(--accent)] active:scale-[0.98] cursor-pointer">
          <Upload className="w-3.5 h-3.5" />
          Upload from device
        </button>
        <button className="px-5 py-2.5 rounded-xl bg-[var(--accent)] text-white text-sm font-semibold transition-all duration-300 hover:bg-[var(--accent-hover)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          Get Started
        </button>
      </div>
    </div>
  );
}
