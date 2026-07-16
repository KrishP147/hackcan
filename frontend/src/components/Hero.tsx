"use client";

import { DropZone } from "./DropZone";
import { RotatingWord } from "./RotatingWord";
import Image from "next/image";

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
        <p className="animate-fade-up stagger-2 mt-7 max-w-5xl text-center text-[clamp(1.2rem,2vw,1.75rem)] font-[500] leading-[1.3] tracking-[0.01em] text-black">
          The Canva for video editing. Make one change and{" "}
          <span className="underline decoration-[var(--accent)] decoration-2 underline-offset-4">
            FrameShift
          </span>{" "}
          carries it through every frame. Your vision. Your changes. Your control.
        </p>
      </div>

      {/* ── Drop zone ── */}
      <div className="animate-fade-up stagger-3 w-full max-w-2xl px-6 pb-12 md:px-12">
        <div className="relative">
          <DropZone />

          <aside
            className="mx-auto mt-12 h-64 w-56 -rotate-[20deg] transition-transform duration-300 hover:rotate-[5deg] hover:scale-[1.02] lg:absolute lg:right-[calc(100%+7.5rem)] lg:-top-8 lg:mt-0"
            aria-label="Photos from Hack Canada"
          >
            <div className="absolute left-0 top-0 w-48 overflow-hidden rounded-[1.5rem] border-4 border-white bg-white shadow-[0_20px_45px_rgba(23,23,23,0.18)]">
              <Image
                src="/hack-canada-3106.jpg"
                alt="Hack Canada event"
                width={1200}
                height={900}
                className="aspect-[4/3] h-auto w-full object-cover"
              />
            </div>
            <div className="absolute left-12 top-16 w-48 overflow-hidden rounded-[1.5rem] border-4 border-white bg-white shadow-[0_20px_45px_rgba(23,23,23,0.22)]">
              <Image
                src="/hack-canada-3131.jpg"
                alt="FrameShift team at Hack Canada"
                width={1200}
                height={900}
                className="aspect-[4/3] h-auto w-full object-cover"
              />
            </div>
            <p className="absolute left-0 top-[14.5rem] w-full text-center text-xs font-semibold leading-4 text-[var(--fg-muted)]">
              FrameShift team
            </p>
          </aside>

          {/* Always visible and outside layout flow on desktop, so it never
              changes the upload rectangle's dimensions or position. */}
          <aside
            className="mx-auto mt-10 w-32 rotate-[20deg] transition-transform duration-300 hover:rotate-[14deg] hover:scale-[1.03] lg:absolute lg:left-[calc(100%+7.5rem)] lg:-top-12 lg:mt-0 lg:w-36"
            aria-label="FrameShift origin story"
          >
            <div className="overflow-hidden rounded-[2rem] border border-black/10 bg-[#a5694d] shadow-[0_24px_60px_rgba(23,23,23,0.18)]">
              <Image
                src="/hackcanadaLogo.png"
                alt="Hack Canada bear mascot"
                width={1553}
                height={1479}
                className="aspect-square h-auto w-full object-cover"
                priority
              />
            </div>
            <p className="mt-6 text-center text-xs font-semibold leading-4 text-[var(--fg-muted)]">
              Originally a Hack Canada project.
            </p>
          </aside>
        </div>
      </div>

    </section>
  );
}
