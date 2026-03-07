"use client";

import { useState, useEffect, useRef } from "react";

const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01";
const TARGET = "anything.";
const PAUSE_MS = 4000;    // time sitting idle between scrambles
const SCRAMBLE_MS = 380;  // how long the chaos phase lasts
const RESOLVE_MS = 55;    // ms between each letter locking in

export function ScrambleWord() {
  const [letters, setLetters] = useState<string[]>(TARGET.split(""));
  const rafRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(
    null as unknown as ReturnType<typeof setTimeout>
  );

  useEffect(() => {
    function randomChar() {
      return CHARS[Math.floor(Math.random() * CHARS.length)];
    }

    function scramble() {
      const start = Date.now();

      function frame() {
        if (Date.now() - start < SCRAMBLE_MS) {
          setLetters(TARGET.split("").map(() => randomChar()));
          rafRef.current = requestAnimationFrame(frame);
        } else {
          resolve(0);
        }
      }

      rafRef.current = requestAnimationFrame(frame);
    }

    function resolve(index: number) {
      if (index >= TARGET.length) {
        setLetters(TARGET.split(""));
        timerRef.current = setTimeout(scramble, PAUSE_MS);
        return;
      }

      setLetters(
        TARGET.split("").map((char, i) =>
          i <= index ? char : randomChar()
        )
      );

      timerRef.current = setTimeout(() => resolve(index + 1), RESOLVE_MS);
    }

    timerRef.current = setTimeout(scramble, PAUSE_MS);

    return () => {
      cancelAnimationFrame(rafRef.current);
      clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <span
      className="font-[550]"
      style={{ color: "var(--accent)" }}
      aria-label={TARGET}
    >
      {letters.join("")}
    </span>
  );
}
