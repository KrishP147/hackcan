const KEYWORDS = ["DETECT", "REMOVE", "RECOLOR", "RESIZE"];

export function ScrollTicker() {
  const items = [...KEYWORDS, ...KEYWORDS, ...KEYWORDS, ...KEYWORDS];

  return (
    <div className="w-full bg-[var(--surface-dark)] py-4 overflow-hidden">
      <div className="animate-scroll-ticker flex whitespace-nowrap">
        {items.map((keyword, i) => (
          <span key={i} className="flex items-center">
            <span className="text-sm font-semibold uppercase tracking-widest text-white/80 px-6">
              {keyword}
            </span>
            <span className="text-[var(--accent)] text-xs">&#9670;</span>
          </span>
        ))}
      </div>
    </div>
  );
}
