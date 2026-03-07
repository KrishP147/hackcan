import { Play } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-[var(--surface-dark)] py-12 px-6 md:px-24">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
            <Play className="w-3.5 h-3.5 text-white ml-0.5" fill="white" />
          </div>
          <span className="text-lg font-[550] text-white tracking-tight">
            ProductName
          </span>
        </div>

        {/* CTA */}
        <button className="px-7 py-3 rounded-xl bg-[var(--accent)] text-white font-semibold transition-all duration-300 hover:bg-[var(--accent-hover)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer">
          Get Started
        </button>
      </div>

      <div className="max-w-7xl mx-auto mt-8 pt-6 border-t border-white/10">
        <p className="text-sm text-[var(--fg-subtle)] text-center md:text-left">
          &copy; 2026 ProductName. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
