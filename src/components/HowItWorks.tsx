"use client";

import { Upload, ScanEye, MousePointerClick, ChevronRight } from "lucide-react";
import { useInView } from "@/hooks/useInView";

const STEPS = [
  {
    number: "01",
    icon: Upload,
    title: "Upload your video",
    description: "Drop a video from your device",
  },
  {
    number: "02",
    icon: ScanEye,
    title: "AI scans every frame",
    description: "Objects, backgrounds, and scenes detected",
  },
  {
    number: "03",
    icon: MousePointerClick,
    title: "Edit anything you want",
    description: "Remove, resize, recolor — just click and edit",
  },
];

export function HowItWorks() {
  const { ref: headingRef, inView: headingInView } = useInView();
  const { ref: stepsRef, inView: stepsInView } = useInView();

  return (
    <section className="bg-[var(--bg-subtle)] py-24 px-6 md:px-12 lg:px-24">
      {/* Heading */}
      <div ref={headingRef} className="text-center mb-16">
        <h2
          className={`font-[550] text-[60px] leading-[60px] text-black ${
            headingInView ? "animate-logo-intro" : "opacity-0"
          }`}
        >
          How it works
        </h2>
      </div>

      {/* Steps */}
      <div
        ref={stepsRef}
        className="max-w-4xl mx-auto flex flex-col md:flex-row items-center md:items-start justify-center gap-8 md:gap-0"
      >
        {STEPS.map((step, i) => (
          <div key={step.number} className="flex items-center md:items-start gap-4 md:gap-0">
            {/* Step card */}
            <div
              className={`text-center flex-1 md:px-6 ${
                stepsInView ? "animate-fade-up" : "opacity-0"
              }`}
              style={{
                animationDelay: stepsInView ? `${(i + 1) * 0.15}s` : undefined,
              }}
            >
              {/* Big number */}
              <span className="text-5xl font-bold text-[var(--accent)] opacity-20 select-none">
                {step.number}
              </span>

              {/* Icon */}
              <div className="flex justify-center mt-3">
                <step.icon className="w-6 h-6 text-[var(--fg)]" strokeWidth={1.5} />
              </div>

              {/* Title */}
              <h3 className="mt-3 text-xl font-[550] text-[var(--fg)]">
                {step.title}
              </h3>

              {/* Description */}
              <p className="mt-2 text-base text-[var(--fg-muted)]">
                {step.description}
              </p>
            </div>

            {/* Arrow between steps (hidden on last step and mobile) */}
            {i < STEPS.length - 1 && (
              <div
                className={`hidden md:flex items-center justify-center shrink-0 mt-12 ${
                  stepsInView ? "animate-fade-up" : "opacity-0"
                }`}
                style={{
                  animationDelay: stepsInView
                    ? `${(i + 1) * 0.15 + 0.1}s`
                    : undefined,
                }}
              >
                <ChevronRight className="w-6 h-6 text-[var(--border)]" />
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
