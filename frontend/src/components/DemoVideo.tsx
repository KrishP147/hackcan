export function DemoVideo() {
  return (
    <section className="w-full px-6 py-16 md:px-12" aria-label="FrameShift product demo">
      <div className="mx-auto max-w-6xl">
        <div className="overflow-hidden rounded-[2rem] border border-black/10 bg-black shadow-[0_28px_80px_rgba(23,23,23,0.16)]">
          <video
            src="/frameshift-demo.mp4"
            controls
            autoPlay
            loop
            muted
            playsInline
            preload="metadata"
            className="block h-auto w-full"
          >
            Your browser does not support HTML video.
          </video>
        </div>
      </div>
    </section>
  );
}
