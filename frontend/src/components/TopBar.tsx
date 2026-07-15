"use client";

import { useEffect, useState } from "react";
import { Play } from "lucide-react";
import { useUser } from "@auth0/nextjs-auth0/client";

export function TopBar() {
  const [scrolled, setScrolled] = useState(false);
  const { user } = useUser();

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
      <div className="max-w-7xl mx-auto px-6 md:px-12 lg:px-24 h-16 flex items-center justify-between">
        <a href="/" className="flex items-center gap-2 animate-logo-intro">
          <div className="w-8 h-8 rounded-full bg-[var(--fg)] flex items-center justify-center transition-colors duration-300 hover:bg-[var(--accent)]">
            <Play className="w-3.5 h-3.5 text-white ml-0.5" fill="white" />
          </div>
          <span className="text-lg font-semibold tracking-tight">
            FrameShift
          </span>
        </a>

        <div className="flex items-center gap-3">
          {user ? (
            <>
              <a
                href="/dashboard"
                className="text-sm font-semibold text-[var(--fg)] hover:text-[var(--accent)] transition-colors"
              >
                Dashboard
              </a>
              {user.picture && (
                <img
                  src={user.picture}
                  alt={user.name ?? ""}
                  className="w-8 h-8 rounded-full object-cover border-2 border-transparent hover:border-[var(--accent)] transition-all cursor-pointer"
                  onClick={() => window.location.href = "/dashboard"}
                />
              )}
              <a
                href="/auth/logout"
                className="text-sm text-[var(--fg-muted)] hover:text-[var(--accent)] transition-colors"
              >
                Sign out
              </a>
            </>
          ) : (
            <a
              href="/auth/login?returnTo=/dashboard"
              className="text-sm font-semibold bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white px-5 py-2 rounded-xl transition-all duration-300 hover:scale-[1.02] active:scale-[0.98]"
            >
              Sign in
            </a>
          )}
        </div>
      </div>
    </header>
  );
}
