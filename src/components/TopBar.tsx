"use client";

import { useEffect, useState } from "react";
import { Play } from "lucide-react";

export function TopBar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/80 backdrop-blur-lg border-b border-gray-100"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 md:px-12 lg:px-24 h-16 flex items-center">
        <a href="/" className="flex items-center gap-2 animate-logo-intro">
          <div className="w-8 h-8 rounded-full bg-[var(--fg)] flex items-center justify-center transition-colors duration-300 hover:bg-[var(--accent)]">
            <Play className="w-3.5 h-3.5 text-white ml-0.5" fill="white" />
          </div>
          <span className="text-lg font-semibold tracking-tight">
            ProductName
          </span>
        </a>
      </div>
    </header>
  );
}
