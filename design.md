# Front-End Design Specification

Cloud-based AI video editor landing page. Inspired by Canva Video Editor's layout structure and Clipabit's typography and animation system. The page starts at the landing, lets users upload or drag-and-drop a video, and transitions into the editor (editor not covered here).

---

## References

- **Layout & structure**: [Canva Video Editor](https://www.canva.com/video-editor/) — bento grids, dual CTAs, demo video with timeline bar, feature cards with distinct backgrounds, sticky bottom CTA
- **Typography & animation**: [Clipabit](https://clipabit.web.app/) — Clash Display variable font, logo-intro animation, scroll ticker, keyword highlight effects, cubic-bezier easing

---

## Global Design Tokens

### Color Palette

| Token              | Value     | Usage                              |
|--------------------|-----------|-------------------------------------|
| `--bg`             | `#FFFFFF` | Page background                     |
| `--bg-subtle`      | `#F9FAFB` | Alternate section background        |
| `--fg`             | `#171717` | Primary text                        |
| `--fg-muted`       | `#6B7280` | Subtitles, descriptions             |
| `--fg-subtle`      | `#9CA3AF` | Card descriptions, placeholders     |
| `--accent`         | `#F43F5E` | Coral/Rose — primary accent         |
| `--accent-hover`   | `#E11D48` | Accent hover state                  |
| `--surface-dark`   | `#111827` | Dark cards, video placeholder       |
| `--surface-darker` | `#1F2937` | Demo placeholder areas inside cards |
| `--border`         | `#D1D5DB` | Drop zone dashed border             |
| `--border-dark`    | `#374151` | Card internal borders               |

No gradients. No purple. Solid colors only.

### Typography

**Font family**: `clashDisplay` variable (WOFF2), fallback: `"clashDisplay Fallback", Arial, Helvetica, sans-serif`

Source: Clash Display from [Fontshare](https://www.fontshare.com/fonts/clash-display) (free for commercial use).

| Role             | Weight | Size                          | Line-Height | Tracking       |
|------------------|--------|-------------------------------|-------------|----------------|
| Hero heading     | 550    | `clamp(2rem, 10vw, 8rem)`    | 1.0         | `tracking-tight` |
| Section heading  | 550    | `60px`                        | `60px`      | `normal`       |
| Card title       | 550    | `24px` (`text-2xl`)           | 1.3         | `normal`       |
| Card label       | 600    | `14px` (`text-sm`)            | 1.0         | `tracking-widest`, uppercase |
| Body / subtitle  | 400    | `20px`                        | `32.5px`    | `normal`       |
| Card description | 400    | `16px` (`text-base`)          | 1.5         | `normal`       |
| Footer / small   | 400    | `14px`                        | `20px`      | `normal`       |
| Button text      | 600    | `16px`                        | 1.0         | `normal`       |

### Spacing

Vertical rhythm between sections: `space-y-24` (96px) — directly from Clipabit.

Inner padding: `p-6` for cards, `p-4` for nested elements, `px-6 md:px-12 lg:px-24` for section containers.

### Border Radius

| Element      | Radius        |
|--------------|---------------|
| Buttons      | `rounded-xl` (12px) |
| Cards        | `rounded-2xl` (16px) |
| Demo frames  | `rounded-xl` (12px) |
| Video player | `rounded-2xl` (16px) |

### Scroll Behavior

```css
html { scroll-behavior: smooth; }
```

---

## Animation System

All animations sourced from or inspired by Clipabit's implementation.

### 1. Logo Intro (Clipabit exact)

Used on: logo, hero heading, section headings on scroll.

```css
@keyframes logo-intro {
  0% {
    opacity: 0;
    filter: blur(2px);
    transform: translateY(-8px) scale(0.92) rotate(-2deg);
  }
  60% {
    opacity: 1;
    filter: blur(0.4px);
    transform: translateY(2px) scale(1.02) rotate(0.5deg);
  }
  100% {
    opacity: 1;
    filter: blur(0);
    transform: translateY(0) scale(1) rotate(0);
  }
}

.logo-intro {
  opacity: 0;
  filter: blur(2px);
  transform: translateY(-8px) scale(0.92) rotate(-2deg);
  animation: logo-intro 0.7s cubic-bezier(0.22, 1, 0.36, 1) 0.12s forwards;
}
```

### 2. Stagger Fade-Up

Used on: hero elements, feature cards, how-it-works steps.

```css
@keyframes fade-up {
  0% {
    opacity: 0;
    transform: translateY(20px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

.fade-up {
  opacity: 0;
  animation: fade-up 0.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

/* Stagger delays for sequential children */
.stagger-1 { animation-delay: 0.2s; }
.stagger-2 { animation-delay: 0.4s; }
.stagger-3 { animation-delay: 0.6s; }
.stagger-4 { animation-delay: 0.8s; }
```

### 3. Scroll Ticker (Clipabit exact)

Used on: feature keyword strip below bento grid.

```css
@keyframes scroll {
  0% { transform: translate(0); }
  100% { transform: translate(-50%); }
}

.animate-scroll {
  will-change: transform;
  backface-visibility: hidden;
  animation: scroll 20s linear infinite;
}
```

### 4. Keyword Highlight (Clipabit)

Used on: accent words in headings ("idea", "anything").

```css
.keyword-highlight {
  display: inline-block;
  font-weight: 700;
  color: var(--accent);
  padding: 0.125rem 0.25rem;
  border-radius: 0.125rem;
  transition: all 500ms ease;
}
```

### 5. Interactive Transitions (Clipabit)

Applied globally to all interactive elements:

```css
/* Buttons */
button, a { transition: all 300ms ease; }

/* Cards */
.feature-card {
  transition: all 300ms ease;
}
.feature-card:hover {
  transform: translateY(-4px);
  border-color: var(--accent);
}
```

### 6. Drop Zone Pulse

Used on: drag-and-drop area during dragover.

```css
@keyframes pulse-border {
  0%, 100% { border-color: var(--accent); opacity: 1; }
  50% { border-color: var(--accent); opacity: 0.5; }
}

.drop-zone-active {
  border-style: solid;
  border-color: var(--accent);
  animation: pulse-border 1.5s ease-in-out infinite;
}
```

### 7. Sticky CTA Fade-In

Used on: bottom CTA bar appearing on scroll past hero.

```css
.sticky-cta {
  opacity: 0;
  transform: translateY(20px);
  transition: all 400ms cubic-bezier(0.22, 1, 0.36, 1);
}
.sticky-cta.visible {
  opacity: 1;
  transform: translateY(0);
}
```

### Scroll-Triggered Animations

All section headings and card groups use Intersection Observer to trigger animations when scrolled into view. Threshold: `0.2`. Animations fire once (no repeat on scroll back up).

---

## Component Specs

### Buttons

**Primary (Get Started)**
- Background: `var(--accent)` / `#F43F5E`
- Text: `#FFFFFF`
- Padding: `12px 28px`
- Font: clashDisplay, 600, 16px
- Border-radius: `rounded-xl` (12px)
- Hover: `var(--accent-hover)` / `#E11D48`, `scale(1.02)`
- Active: `scale(0.98)`

**Secondary (Upload from device)**
- Background: transparent
- Border: `1px solid #171717`
- Text: `#171717`
- Same padding, radius, font as primary
- Hover: `bg-#F9FAFB`, `border-color: var(--accent)`

### Drop Zone

- Border: `2px dashed var(--border)`
- Border-radius: `rounded-2xl`
- Padding: `48px`
- Background: transparent
- Text: clashDisplay 400, `#6B7280`, centered
- "or" divider between text and buttons
- **Drag hover state**: border becomes `solid var(--accent)`, pulse animation, background `rgba(244, 63, 94, 0.04)`

### Demo Video Placeholder

- Aspect ratio: 16:9
- Background: `var(--surface-dark)` / `#111827`
- Border-radius: `rounded-2xl`
- Centered play button: 64px circle, `rgba(255,255,255,0.15)` bg, white triangle icon
- Play button hover: `rgba(255,255,255,0.25)`, `scale(1.1)`, `transition-all 300ms`

### Timeline Bar (Canva-inspired)

Thin bar (6px height) directly below the video placeholder, `rounded-full`, composed of colored segments:

| Segment | Color     | Width |
|---------|-----------|-------|
| 1       | `#F43F5E` | 30%   |
| 2       | `#F59E0B` | 25%   |
| 3       | `#0EA5E9` | 20%   |
| 4       | `#10B981` | 25%   |

This gives a visual hint of a video timeline without any actual video content.

### Feature Card

- Background: varies per card (see Section 2 layout)
- Border-radius: `rounded-2xl`
- Padding: `p-6`
- Category label: top-left, coral, uppercase, tracking-widest, text-sm, weight 600
- Demo area: `rounded-xl`, `var(--surface-darker)`, `aspect-video`, dashed border `var(--border-dark)`
- Title: white, weight 550, text-2xl
- Description: `var(--fg-subtle)`, weight 400, text-base
- Hover: `translateY(-4px)`, `border 1px solid var(--accent)`, `transition-all 300ms`

### Scroll Ticker Strip

Horizontal infinite-scroll bar of keywords, directly below the bento grid:

```
▬ DETECT ▬ REMOVE ▬ RECOLOR ▬ RESIZE ▬ DETECT ▬ REMOVE ▬ ...
```

- Background: `var(--surface-dark)`
- Text: clashDisplay 600, 14px, `tracking-widest`, uppercase
- Separator dots: `var(--accent)`
- Full width, `overflow: hidden`
- Content duplicated for seamless loop
- Animation: `scroll 20s linear infinite`

---

## Page Layout

### Section 1: Hero (full viewport)

```
┌──────────────────────────────────────────────────────────┐
│ [Logo] ProductName                                        │
│                                                           │
│                                                           │
│         Edit your videos                                  │
│           with just an idea.                              │
│                                                           │
│         AI-powered editing. No complexity.                │
│                                                           │
│    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐      │
│    │      Drag and drop your video here            │      │
│    │                  or                           │      │
│    │    [Upload from device]    [Get Started]      │      │
│    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘      │
│                                                           │
│    ┌──────────────────────────────────────────────┐       │
│    │                                              │       │
│    │              ▶  Demo Video                   │       │
│    │                                              │       │
│    │  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬  │       │
│    └──────────────────────────────────────────────┘       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Background**: `var(--bg)` / `#FFFFFF`

**Top bar**: fixed, transparent initially. On scroll: white bg, `backdrop-blur-lg`, subtle `border-b border-gray-100`. Contains only logo on left. No nav links — the page is action-first.

**Heading**: Clash Display, weight 550, `text-[clamp(2rem,10vw,8rem)]`, `tracking-tight`, `leading-none`, `#171717`. The word "idea" in `var(--accent)` with the keyword-highlight treatment.

**Subtitle**: Clash Display, weight 400, 20px, `#6B7280`. Below the heading with `mt-6`.

**Drop zone**: centered, `max-w-2xl`, `mt-10`. Contains drag-drop text, "or" divider, and the two CTA buttons side by side.

**Demo video placeholder**: centered, `max-w-4xl`, `mt-12`. 16:9 dark frame with play button. Colored timeline bar directly underneath.

**Entrance animation sequence**:
1. Logo: `logo-intro` (0.7s, 0.12s delay)
2. Heading: `fade-up` (0.6s, 0.2s delay)
3. Subtitle: `fade-up` (0.6s, 0.4s delay)
4. Drop zone: `fade-up` (0.6s, 0.6s delay)
5. Video placeholder: `fade-up` (0.6s, 0.8s delay)

All eased with `cubic-bezier(0.22, 1, 0.36, 1)`.

---

### Section 2: Feature Bento Grid

**Background**: `var(--bg)` / `#FFFFFF`

**Heading**: "Edit anything in frame." — Clipabit big tagline style: `text-[clamp(2rem,10vw,8rem)]`, `font-extrabold`, `tracking-tight`, `leading-none`, centered. "anything" in `var(--accent)` with keyword-highlight. Triggered by Intersection Observer with `logo-intro` animation.

**Grid**: 2 columns, `gap-4`, `max-w-5xl`, centered.

| Card | Label   | Title               | Description                              | Background |
|------|---------|---------------------|------------------------------------------|------------|
| 1    | DETECT  | AI Object Detection | Select any object just by clicking on it | `#111827`  |
| 2    | REMOVE  | Background Removal  | Remove or swap backgrounds instantly     | `#18181B`  |
| 3    | RECOLOR | Color Grading       | Change the color of any detected object  | `#1C1917`  |
| 4    | RESIZE  | Smart Resize        | Reframe for any platform in one click    | `#F43F5E`  |

Card 4 uses the coral accent as its background with white text throughout — creates visual asymmetry and draws attention (Canva uses this pattern with distinct card colors).

Each card contains a demo image placeholder area: `aspect-video`, `rounded-xl`, slightly lighter than card bg with dashed border.

Cards stagger in on scroll: `fade-up`, 100ms delay between each.

**Scroll ticker strip**: full-width bar below the grid. `var(--surface-dark)` bg, uppercase keywords separated by coral dots, scrolling infinitely left.

---

### Section 3: How It Works

**Background**: `var(--bg-subtle)` / `#F9FAFB`

**Heading**: "How it works" — Clash Display, weight 550, 60px, centered, `#000000`. Triggers with `logo-intro` on scroll.

**Layout**: 3 columns, equal width, centered, `max-w-4xl`.

```
     01               02               03
   [icon]    ──→    [icon]    ──→    [icon]
   Upload          AI scans          Edit
  your video      every frame     anything you want

  Drop a video    Objects, back-    Remove, resize,
  from your       grounds, and      recolor — just
  device          scenes detected   click and edit
```

**Step numbers**: Clash Display, weight 700, `text-5xl`, `var(--accent)`, `opacity-20` — large background numbers for visual weight.

**Icons**: simple line icons (24px stroke, `#171717`), centered above step title. Upload arrow, scan/eye, edit/cursor.

**Step title**: Clash Display, weight 550, `text-xl`, `#171717`.

**Step description**: Clash Display, weight 400, `text-base`, `var(--fg-muted)`.

**Arrows**: `#D1D5DB` chevrons between steps. Hidden on mobile (steps stack vertically).

**Animation**: steps stagger left-to-right on scroll, `fade-up`, 150ms gaps. Arrows animate in 100ms after their preceding step.

---

### Section 4: Footer

**Background**: `var(--surface-dark)` / `#111827`

**Layout**: simple flex row, `justify-between`, `items-center`, `py-12 px-6 md:px-24`.

Left: `[Logo] ProductName` in white, Clash Display weight 550.

Right: "Get Started" button (coral, same primary button spec).

Below: `© 2026 ProductName. All rights reserved.` — Clash Display 400, 14px, `var(--fg-subtle)`.

---

### Sticky Bottom CTA Bar (Canva pattern)

**Visibility**: appears when user scrolls past the hero's drop zone (use Intersection Observer on drop zone — when it exits viewport, show bar).

**Layout**: fixed bottom, full width, `py-3 px-6`, white bg, `backdrop-blur-lg`, `border-t border-gray-100`, `z-50`.

**Content**: centered row with "Upload from device" (secondary) + "Get Started" (primary) buttons.

**Animation**: `sticky-cta` class — `opacity 0→1`, `translateY(20px→0)`, `400ms cubic-bezier(0.22, 1, 0.36, 1)`.

---

## Responsive Behavior

### Mobile (< 768px)

- Hero heading: `clamp` scales down automatically to `2rem`
- Drop zone: full width, buttons stack vertically
- Bento grid: single column
- How It Works: steps stack vertically, arrows hidden
- Sticky CTA: buttons shrink, `text-sm`, less padding
- Scroll ticker: same, just faster (`12s` instead of `20s`)

### Tablet (768px - 1024px)

- Bento grid: 2 columns maintained, smaller gap
- How It Works: 3 columns maintained with smaller text
- Max-width containers adjust to `max-w-3xl`

### Desktop (> 1024px)

- Full layout as designed
- Max-width containers at `max-w-5xl` for grid, `max-w-4xl` for how-it-works
- Generous whitespace maintained

---

## Recommendations

1. **Logo placeholder**: simple geometric mark — circle with play-triangle cutout, similar to Clipabit's logo style. Monochrome `#171717`, coral on hover.

2. **Micro-interactions**: every clickable element should have `transition-all 300ms`. No element should feel static. Buttons scale on press (`active:scale-98`), cards lift on hover, the drop zone border animates.

3. **No skeleton screens needed** for landing page — all content is static. The demo placeholders are intentionally styled as placeholders (dark frames with dashed borders).

4. **Performance**: load Clash Display via `@font-face` with `font-display: swap` and preload the WOFF2. Keep animations GPU-accelerated (`will-change: transform, opacity`). Use `contain: layout style paint` on scroll ticker.

5. **Accessibility**: all animations respect `prefers-reduced-motion` — disable keyframe animations, keep transitions at 0ms. Drop zone supports keyboard (Enter/Space triggers file picker). Color contrast ratios: white on `#111827` = 15.4:1 (AAA), `#6B7280` on white = 4.6:1 (AA).

6. **Anti-vibe-code signals**: consistent spacing (multiples of 4px via Tailwind), single font family throughout, restrained color palette (only one accent), no decorative SVG blobs, no floating particles, no glassmorphism. The design reads as intentional because every element has a clear purpose.

---

## Tech Stack Suggestion

- **Framework**: Next.js (App Router) or Vite + React
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui (already in package.json as devDependency)
- **Animations**: CSS keyframes + Intersection Observer (no heavy library needed)
- **Font**: Self-hosted Clash Display WOFF2
- **Icons**: Lucide React (pairs well with shadcn)
