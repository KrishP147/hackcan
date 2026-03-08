export function DemoVideo() {
  return (
    <div className="max-w-4xl mx-auto px-4">
      <div className="relative aspect-video bg-[var(--surface-dark)] rounded-2xl overflow-hidden">
        <video
          src="/FrameShift.mp4"
          controls
          playsInline
          className="w-full h-full object-contain"
          poster="/Thumbnail.png"
        >
          Your browser does not support the video tag.
        </video>
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
