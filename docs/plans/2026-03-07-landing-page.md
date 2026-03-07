# Landing Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the landing page for the AI video editor per design.md spec.

**Architecture:** Next.js App Router with Tailwind CSS v4, Clash Display variable font via next/font/local, shadcn/ui components where useful. Four sections: Hero, Feature Bento Grid, How It Works, Footer. Plus a sticky CTA bar.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, Lucide React, next/font/local (Clash Display WOFF2)

---

### Task 1: Initialize Next.js project

**Step 1:** Scaffold Next.js with create-next-app (TypeScript, Tailwind, App Router, src dir)
**Step 2:** Install lucide-react
**Step 3:** Verify dev server starts
**Step 4:** Commit

### Task 2: Set up Clash Display font + global CSS

**Step 1:** Download Clash Display variable WOFF2 into src/fonts/
**Step 2:** Configure next/font/local in layout.tsx with CSS variable --font-clash
**Step 3:** Set up global CSS: design tokens (colors), animation keyframes (logo-intro, fade-up, scroll, pulse-border), base styles
**Step 4:** Configure Tailwind to use Clash Display font family
**Step 5:** Commit

### Task 3: Build Hero section

**Step 1:** Create components: TopBar, HeroHeading, DropZone, DemoVideo, TimelineBar
**Step 2:** Assemble in page.tsx as full-viewport hero
**Step 3:** Wire up drag-and-drop zone with dragover/dragleave states
**Step 4:** Add staggered entrance animations
**Step 5:** Commit

### Task 4: Build Feature Bento Grid + Scroll Ticker

**Step 1:** Create FeatureCard component with props (label, title, description, bg)
**Step 2:** Create BentoGrid section with 4 cards per design spec
**Step 3:** Create ScrollTicker component with infinite horizontal animation
**Step 4:** Add scroll-triggered animations via Intersection Observer hook
**Step 5:** Commit

### Task 5: Build How It Works section

**Step 1:** Create StepCard component (number, icon, title, description)
**Step 2:** Create HowItWorks section with 3 steps + arrow connectors
**Step 3:** Add scroll-triggered stagger animations
**Step 4:** Commit

### Task 6: Build Footer + Sticky CTA

**Step 1:** Create Footer component (dark bg, logo, CTA, copyright)
**Step 2:** Create StickyCTA component (appears on scroll past hero drop zone)
**Step 3:** Wire up Intersection Observer for sticky CTA visibility
**Step 4:** Commit

### Task 7: Responsive + reduced-motion + final polish

**Step 1:** Test and fix mobile layout (single column grid, stacked buttons, hidden arrows)
**Step 2:** Add prefers-reduced-motion media query
**Step 3:** Final commit
