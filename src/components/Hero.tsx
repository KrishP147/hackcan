"use client";

import { DropZone } from "./DropZone";
import { DemoVideo } from "./DemoVideo";
import { RotatingWord } from "./RotatingWord";

export function Hero() {
  return (
    <section className="flex flex-col items-center text-center">

      {/* ── Above-fold content ── */}
      <div className="w-full max-w-4xl px-6 pt-24 pb-6 flex flex-col items-center">

        {/* Heading */}
        <h1 className="animate-fade-up stagger-1 font-[550] tracking-tight leading-[1.15] text-[clamp(2rem,5.5vw,4.5rem)]">
          One click.
          <br />
          <RotatingWord />
          <br />
          <span style={{ color: "var(--accent)" }}>anything.</span>
        </h1>

        {/* Subtitle */}
        <p className="animate-fade-up stagger-2 mt-4 text-[var(--fg-muted)] text-base max-w-md text-center">
          AI detects every object in your video. Click to remove, resize, or
          recolor — no expertise required.
        </p>
      </div>

      {/* ── Drop zone ── */}
      <div className="animate-fade-up stagger-3 w-full px-6 md:px-12 max-w-2xl pb-10">
        <DropZone />
      </div>

      {/* ── Large demo video — top visible at page load ── */}
      <div className="w-full px-4 md:px-6 lg:px-8 pb-16">
        <DemoVideo />
      </div>

    </section>
  );
}
