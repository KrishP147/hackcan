# FrameShift — Production Pipeline (Document 1 of 2)

The complete shipping system: all editing tools working cleanly, object-morphing solved. **This entire path runs on torch built-ins** (`grid_sample`, RAFT from torchvision) and OpenCV — **zero custom-build dependency**, deploys on Modal today. The optional performance/quality upgrade (guided-synthesis CUDA kernel) lives in Document 2 and is never on this critical path.

**The decoupling that governs both docs:** production ships on torch; the hand-written CUDA kernel is a parallel track guarded behind a flag so neither goal holds the other hostage. Backend A (this doc) is always the fallback for Backend B (Doc 2).

**Bar for "working":** for each tool, play a full clip and watch it — no morph, no flicker, no seam crawl, no cut line.

---

## 0. The one organizing insight

Morphing is **only a generative-path problem.** It happened because `replace` generated the object independently at several keyframes and RIFE blended between mismatched draws. Every *deterministic* tool (recolor, blur, resize, delete, color_pop, glow) is a per-frame function of the original pixels and the tracked mask — temporally consistent **by construction**, given a stable mask and coherently-filled reveals. That splits the system cleanly:

- **Deterministic tools** need only: (a) stable masks (§2.7), (b) the temporally-coherent inpainter (§4) *where they reveal hidden background*.
- **`replace`** (single generated appearance to carry) needs the full flow-propagation engine (§2) — the morphing fix — with an optional Ezsynth-inspired synthesis backend (§3, kernel in Doc 2).
- **`background_replace`** needs a soft matte + a stabilized new-background plate (§5.5).

Don't over-engineer the deterministic set. Concentrate the hard machinery where it's needed.

---

## 1. Architecture overview

```
upload ──ffmpeg──> frames/ (jpg) ──SAM2──> masks/ (png, per tracked object)
                          │
                          ├── RAFT (pairwise flow, cached both directions)      [§2.1]
                          │
edit request ──dispatch──┤
                          ├─ deterministic tool ─> per-frame op(mask, original) ─┐
                          │                          + inpaint reveals (§4)       │
                          ├─ replace ─> PROPAGATION ENGINE (§2/§3) + reanchor ────┤─> composite ─> render
                          └─ background_replace ─> matte + bg plate (§5.5) ───────┘
```

**Propagation engine interface (one signature, two backends):**

```python
def propagate(anchor_edit, frames, masks, flows, anchors) -> list[EditLayer]:
    # Backend A (default, ships): flow star-warp on grid_sample   §2
    # Backend B (optional):       guided synthesis                §3  (CUDA kernel = Document 2)
```

Both obey the same discipline — **carry one appearance, sample the pristine source, never chain off your own output** — and are hot-swappable behind a config flag. Backend A is torch-native and deploys with zero custom build; Backend B is the Ezsynth-inspired upgrade for heavy stylization and disocclusion.

Modules: `services/flow_propagation_service.py` (Backend A + shared engine); `services/synth_propagation_service.py` (Backend B, optional).

---

## 2. Shared flow engine (Backend A + the morphing fix)

All heavy math in torch on GPU (MPS locally, CUDA on Modal).

### 2.1 RAFT — optical flow

`torchvision.models.optical_flow.raft_large` (torchvision 0.25.0 in the venv — no new dependency). RAFT builds a 4D all-pairs correlation volume between per-pixel features of `I_t, I_{t+1}`, then a conv-GRU iteratively refines flow (12–32 iterations) at 1/8 resolution with learned convex upsampling to full res.

- Inputs normalized to `[-1,1]`, `H,W` divisible by 8 (720p = 1280×720 satisfies this).
- `num_flow_updates=12`, fp16; ~40ms/pair on a T4. A 300-frame clip ≈ 600 flow computations (fwd+bwd) ≈ 25s. **Cache to disk** `flows/flow_{fwd,bwd}_%04d.npy`. Flow is computed on **original footage** → edit-independent, cached once per project, reused by every tool and re-edit.

### 2.2 Star warp — the morphing fix (carry, don't re-invent)

Gemini generates the edit **once** at anchor `A`. Define edit layer `E_A` = edited RGBA cutout inside `M_A`.

Do **not** chain a warp off each previous output (that stacks one bilinear blur per frame). **Compose the pairwise flow into a single `A→t` displacement** and take exactly one clean sample of the pristine `E_A`:

```
F_{N→N} = 0
F_{N→k-1}(x) = F_{N→k}(x) + sample( F_{k→k-1}, x + F_{N→k}(x) )   # compose flow: warp a smooth vector field
E_N(x)      = sample( E_A, x + F_{N→A}(x) )                       # ONE resample of sharp RGBA
```

Each composition resamples a *flow field* (smooth → negligible blur) and accumulates only geometric drift (capped by re-anchoring, §2.6). The single RGBA resample is the only place image blur enters, off the pristine anchor. One appearance, transported along true scene motion → nothing to morph between.

Backward sampling (`grid_sample`, not forward splatting) gives every target pixel exactly one well-defined source — no holes or collisions.

```python
import torch, torch.nn.functional as F
def warp(img, flow):                       # img (N,C,H,W), flow (N,2,H,W) px
    N,_,H,W = img.shape
    yy,xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    base  = torch.stack((xx,yy)).float().to(img)
    src   = base[None] + flow
    gx = 2*(src[:,0]+0.5)/W - 1; gy = 2*(src[:,1]+0.5)/H - 1
    grid = torch.stack((gx,gy),-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
```

Pin the `align_corners=False` / pixel-center (`+0.5`) convention **here, once** — Doc 2's kernels must match it byte-for-byte or validation diffs won't be zero.

Warp a **validity channel** `V ∈ [0,1]` alongside RGBA (init 1 at anchor), decaying where sampling pulls out-of-bounds or occluded → per-pixel confidence for free.

### 2.3 Forward-backward occlusion gate

On the composed `A→N` vs `N→A` flow (Sundaram et al.):

```
err(x)      = ‖ F_fwd(x) + sample(F_bwd, x + F_fwd(x)) ‖²
occluded(x) = err(x) > 0.01·(‖F_fwd(x)‖² + ‖F_bwd(x)‖²) + 0.5
```

A pixel that doesn't round-trip → **zero its validity** = a hole to fill, not a smear to propagate. Stops the warp dragging wrong pixels across an occlusion boundary. Hole policy: inside `M_t` with `V_t < 0.5` → fill from bidirectional partner (§2.4), else inpainter (§4).

### 2.4 Bidirectional propagation (production default)

Star-warp **twice** — nearest **past** and nearest **future** anchor — blend per pixel by temporal distance × validity:

```
w_past(x) = V_past(x) / (dist_past + ε)
w_fut(x)  = V_fut(x)  / (dist_fut  + ε)
E_t(x)    = (w_past·E_past(x) + w_fut·E_fut(x)) / (w_past + w_fut)
```

Fills disocclusions (hidden forward, visible backward) and reduces drift (averaging two carries). Both invalid → inpainter (§4). Needs a future anchor: produce the end-of-range anchor via warp-then-refine (§2.6), then propagate inward from both ends.

### 2.5 Compositing

```
out_t = α_t ⊙ E_t + (1 − α_t) ⊙ I_t
```

`α_t` = mask `M_t` feathered 5px (hard edges leave a visible cut line). Background stays original footage — temporally coherent for free.

### 2.6 Warp-then-refine re-anchor (never regenerate from scratch)

When the warp can't carry further — newly revealed surfaces, large perspective change, confidence collapse — do **not** regenerate independently (that's a second draw = morphing). Seed generation **from the warped frame** as an img2img init at **low denoise strength** so the model cleans up rather than reinvents:

```python
gemini_service.edit_frame_with_reference(
    warped_frame_t, prompt="clean up warp artifacts; preserve this exact appearance",
    reference_frame_path=anchor_A, mask_path=M_t)   # already exists
```

Triggers (any fires): mean `V` inside `M_t` < 0.7; mask-area drift `|area(M_t)/area(M_A) − 1| > 0.3`; or 60 frames (~2s) since last anchor. Each new anchor is a fresh past/future endpoint for §2.4. Strongest consistency control the Gemini API exposes; no self-hosted fallback in scope.

### 2.7 Mask conditioning (all tools)

SAM 2 masks jitter a pixel or two at boundaries — invisible for blur, visible for glow/color-pop. Fix: morphological close (3×3) + 2-frame EMA on the mask alpha. Matte-critical tools (color_pop, background_replace) upgrade to a soft matte (§5.5).

---

## 3. Ezsynth-inspired guided propagation (Backend B — quality ceiling)

Flow warp is fast and ships clean but has two limits: it *resamples* rather than *re-synthesizes* (heavy stylization can soften) and can only carry pixels the flow reaches (disocclusions → holes). Ezsynth's frame-propagation technique — guided PatchMatch synthesis — addresses both. **The CUDA kernel powering it is Document 2.** This section is the interface Backend B must satisfy.

### 3.1 Why patch synthesis, conceptually

Ezsynth never resamples its own output. Every target frame is **re-synthesized from patches of the pristine edited keyframe**, guided into place by correspondence. Sharp source always → no chain blur; best-match *search* → graceful degradation at disocclusion and under appearance change. Same "sample from source, not chain" discipline as the star warp, via a nearest-neighbor patch field instead of one flow warp.

### 3.2 The guides (built on Backend A's flow)

| Guide | Source | Target | Purpose |
|---|---|---|---|
| appearance | original anchor `S'` | original target `T'` | primary correspondence (match here, copy edited pixels) |
| edge | edge map of `S'` | edge map of `T'` | keep silhouette/structure aligned |
| positional | identity coords | coords advected by RAFT flow | seed where patches land |
| temporal | prev synth frame | prev synth advected by flow (+valid mask) | frame-to-frame coherence |

Match is in the *original/guide* domain; copied pixels come from the *edited* anchor — that's what makes it an edit propagator, not a style-averager.

### 3.3 Interface (kernel-agnostic)

```python
# synth_propagation_service.py — same signature as Backend A
def propagate(anchor_edit, frames, masks, flows, anchors) -> list[EditLayer]:
    guides = build_guides(frames, flows)                 # appearance, edge, positional, temporal
    return ebsynth_synthesize(anchor_edit, guides, masks)  # CPU ref OR CUDA kernel (Doc 2)
```

`ebsynth_synthesize` = CPU reference (pip `ebsynth`/Ezsynth) or the custom CUDA kernel (Doc 2), chosen by flag. Output shape matches Backend A so §2.5 composite is unchanged.

### 3.4 Backend selection policy

| Situation | Backend |
|---|---|
| `replace`, mild appearance change, mostly translational motion | A (flow warp) |
| `replace`, heavy stylization / texture / large shape change | B (guided synthesis) |
| Long clips, frequent disocclusion | B, or A + aggressive bidirectional + inpaint |
| Deploy where build risk matters | A only |

Ship A first; light up B once the kernel or CPU ref clears the per-tool bar. **Deployment never depends on the CUDA kernel compiling.**

---

## 4. Temporally-coherent inpainting (shared hard component)

Needed by `delete`, `move`, `resize`-down, and disocclusions in `replace`/`background_replace`. Per-frame OpenCV TELEA has no temporal memory → flicker. Build in ascending order; ship the first that clears the bar.

1. **Flow-guided fill (first).** For a hole in frame `t`, pull background from neighboring frames where those pixels *were* visible, warped in via cached RAFT flow (both directions). Most vacated/revealed background is visible in some nearby frame → sourced, not re-guessed → most flicker gone. Confidence = flow validity; blend donors by temporal distance.
2. **TELEA residual.** Truly-never-seen pixels (small) → OpenCV TELEA, then EMA across frames so the patch doesn't shimmer. TELEA's good regime.
3. **Learned escalation.** Complex backgrounds → flow-based video-inpainting network (ProPainter-class) behind the same interface. Heavier dep; only if needed. (An EbSynth-style patch inpaint is also viable and reuses Doc 2's kernel.)

```python
def inpaint_video(frames, hole_masks, flows) -> filled_frames: ...
```

`delete`, `move`, `resize`, generative disocclusion all call this identically.

---

## 5. Per-tool implementation (all tools)

Each tool = **source `S`**, **mask `α`**, **base plate `B`**, **hole strategy**, **backend**. Composite always `out = α·S + (1−α)·B`.

### 5.1 Deterministic in-mask — recolor, blur, color_pop, glow

No generation, no propagated layer. Apply the OpenCV op per frame against the stabilized mask; composite over original. Consistent by construction.

- **recolor** — HSV/LAB shift or LUT on `M_t`. `S`=recolored original, `α`=`M_t`, `B`=original.
- **blur** — Gaussian/bokeh on `M_t`; feather so it doesn't hard-edge.
- **color_pop** — desaturate *outside* the object. `α`=`1−M_t`, `S`=grayscale original, `B`=original. Inverse mask → use soft matte (§5.5) so the boundary doesn't crawl. No new pixels → no inpaint.
- **glow** — bloom halo; composite object over a blurred bright copy; halo extends outside `M_t` so feather the falloff.

Acceptance: masks stabilized; edges don't crawl; no inpaint.

### 5.2 resize (reveals background when shrinking)

Scale the cutout about its centroid.
- **Up** — supplies its own pixels, occludes background (fine). Edge feather only.
- **Down** — reveals a ring of background → **inpaint** via §4. `S`=scaled cutout, `α`=scaled mask, `B`=inpainted original.

Deterministic (cutout from original frame `t`) → no morph; only the reveal needs the inpainter.

### 5.3 delete

`α`=0 in the object region; whole `M_t` becomes a hole → **§4 inpainter** sources vacated background from neighbors. `move`'s background half without a paste. Hardest when the background is never fully visible anywhere → escalate §4.3.

### 5.4 move

1. **Object transport.** Cut via `M_t` (from original → deterministic). Place at interpolated offset `(dx,dy)_t` with feathered alpha (or star-warp if it should also follow scene motion).
2. **Vacated background.** Fill via **§4 inpainter**, hole = original `M_t` minus the new footprint.

Spill handled: moved object's mask is the *moved* mask `M_t+δ` so the stencil matches current content. Acceptance: object crisp along the path; vacated fill flicker-free; no double-image.

### 5.5 background_replace (matte-critical)

1. **Soft matte.** Hard mask cut-lines against a new background → upgrade `M_t` to a soft matte (guided filter or matting model on the SAM 2 trimap) so hair/motion-blur edges composite seam-free. Matte quality is the whole ballgame.
2. **New background plate.** Static or generated; stabilize to camera motion via *scene* flow (inverse of object flow) so it tracks, not slides. Generated bg → generate once, carry like an edit layer (same anti-morph discipline).
3. **Composite.** `out = matte·object + (1−matte)·new_bg`.

Acceptance: no edge fringe; bg tracks camera; edge stable.

### 5.6 replace (the generative object — where morphing lived)

Full engine: generate once (§2.2) → bidirectional propagate (§2.4) + occlusion gate (§2.3) → warp-then-refine re-anchor (§2.6) → composite (§2.5). Backend A default; Backend B (§3) for heavy stylization. Disocclusions inside the object → §4 within the edit layer only. Acceptance: appearance stable across the clip (no morph/ghost/smear); silhouette tracks; re-anchors invisible.

---

## 6. Dispatch & data model

```
recolor, blur, color_pop, glow   -> deterministic per-frame (§5.1)
resize                           -> deterministic + inpaint reveal (§5.2)
delete                           -> inpaint region (§5.3)
move                             -> transport + inpaint vacated (§5.4)
background_replace               -> matte + bg plate (§5.5)
replace                          -> propagation engine + reanchor (§5.6)
```

Storage: `projects/{id}/frames/frame_%04d.jpg`, `masks/mask_%04d.png` (0/255), `flows/flow_{fwd,bwd}_%04d.npy`, status JSON via `project_manager`. `EditRule` carries `type`, object id, per-keyframe params, `backend` override.

---

## 7. Upload / extraction

- `ffprobe` real fps; extract at native rate (drop forced `-vf fps=30`).
- Working frames `scale=-2:720`; keep originals for final render. `/render` re-applies edit ops at full res only for edited ranges; 720p flows/masks upsample cleanly.
- GPU encode (NVENC) where available. Drop per-frame Supabase uploads. Clip cap 15s. Target upload→editor < 10s.

---

## 8. Removals

- `main.py`: delete `/ai/edit/*` (~804–992), `/edit/refine` (~630), `_background_ai_edit` (~1195), commented `/detect`; replace `_background_propagate_changes` with §6 dispatch.
- Delete `services/yolo_service.py` + `yolo11n.pt`; drop `ultralytics` (cold-start win).
- `rife_service.py` + `rife_vendor/` stay for slow-mo only, **off the edit path** (RIFE never touches object propagation — the morphing source).
- Frontend: remove AI chat pane, Enhance/Upscale/Restore.

---

## 9. Deploy

- Modal: FastAPI as `@modal.asgi_app()`. Image: `debian-slim` + torch/torchvision `cu121` + sam2 + opencv-headless + ffmpeg (+ matting/inpaint weights if used). **No custom native build on this path** — RAFT ships pre-compiled; Doc 2's kernel is optional and off this path.
- SAM 2 + RAFT (+ matting/inpaint) in `@modal.enter()` with memory snapshots → warm cold starts.
- GPU: A10G for production (headroom for RAFT + SAM 2 + matting + inpaint per clip); T4 if profiling allows.
- `modal.Volume` at projects dir; path-based `project_manager` unchanged. Secrets: Gemini + Cloudinary. CORS pinned to Vercel. Per-IP token-bucket rate limit. Auth0 on.
- Vercel: frontend as-is; `NEXT_PUBLIC_API_URL` → Modal.

---

## 10. Schedule (production)

1. Shared engine (RAFT+cache, star warp, gate, bidirectional, composite, mask cond.) — 3–4 d
2. Deterministic family (recolor, blur, color_pop, glow) — 2 d
3. Deploy to Modal, lock green — 2–3 d
4. Temporally-coherent inpainter (§4) — 3–4 d
5. delete + resize + move (ride the inpainter) — 3–4 d
6. replace (engine + warp-then-refine, Backend A) — 2–3 d
7. background_replace (matte + bg stabilization) — 3–4 d
8. **Backend B (Doc 2)** — CPU ref then CUDA kernel; parallel track, off critical path — 4–8 d
9. Hardening (re-anchor tuning, full-res render, per-tool acceptance) — 2–3 d

Core (1–7, 9): ~18–24 working days. Backend B/CUDA parallel, never on the critical path.

---

## 11. Acceptance criteria (per tool — "play it and watch it")

| Tool | Passes when… |
|---|---|
| recolor / blur / glow | edge doesn't crawl; effect stable; no cut line |
| color_pop | color/gray boundary clean and stable (soft matte) |
| resize | no seam; shrink reveals clean, flicker-free background |
| delete | vacated region flicker-free, no crawling texture |
| move | object crisp along path; vacated fill clean; no double-image |
| background_replace | no edge fringe; bg tracks camera; edge stable |
| replace | appearance stable across clip (no morph/ghost); re-anchors invisible |

The unit is the clip, never a single frame.
