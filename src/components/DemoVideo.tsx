import { Play } from "lucide-react";

export function DemoVideo() {
  return (
    <div className="max-w-[1440px] mx-auto">
      {/* Video Frame */}
      <div className="relative aspect-video bg-[var(--surface-dark)] rounded-2xl overflow-hidden">
        <button
          className="absolute inset-0 flex items-center justify-center group cursor-pointer"
          aria-label="Play demo video"
        >
          <div className="w-16 h-16 rounded-full bg-white/15 flex items-center justify-center transition-all duration-300 group-hover:bg-white/25 group-hover:scale-110">
            <Play className="w-6 h-6 text-white ml-1" fill="white" />
          </div>
        </button>
      </div>

      {/* Timeline Bar */}
      <div className="flex h-1.5 mt-3 rounded-full overflow-hidden gap-0.5">
        <div className="bg-[#F43F5E] rounded-full" style={{ width: "30%" }} />
        <div className="bg-[#F59E0B] rounded-full" style={{ width: "25%" }} />
        <div className="bg-[#0EA5E9] rounded-full" style={{ width: "20%" }} />
        <div className="bg-[#10B981] rounded-full" style={{ width: "25%" }} />
      </div>
    </div>
  );
}
