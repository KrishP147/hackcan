"use client";

import { useState, useEffect } from "react";

const WORDS = ["Remove", "Resize", "Recolor", "Detect"];

type WordConfig = { color: string; innerClass: string };

const WORD_STYLE: Record<string, WordConfig> = {
  Remove:  { color: "#F43F5E", innerClass: "word-anim-remove"  },
  Resize:  { color: "#10B981", innerClass: "word-anim-resize"  },
  Recolor: { color: "#F59E0B", innerClass: "word-anim-recolor" },
  Detect:  { color: "#F97316", innerClass: "word-anim-detect"  },
};

const INTERVAL_MS = 3000;

export function RotatingWord() {
  const [index, setIndex] = useState(0);
  const [animKey, setAnimKey] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % WORDS.length);
      setAnimKey((prev) => prev + 1);
    }, INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  const word = WORDS[index];
  const { color, innerClass } = WORD_STYLE[word];

  return (
    <span key={animKey} className="rotating-word" style={{ color }}>
      {/* Inner span carries the per-word continuous animation,
          separate from the word-cycle entrance on the outer span */}
      <span className={innerClass}>{word}</span>
    </span>
  );
}
