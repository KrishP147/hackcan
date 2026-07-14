# FrameShift — Technical Overview with Deep Dives

A quick overview of FrameShift, the main design decisions behind it, and how those choices affected the final result. Along the way this drops down into how each underlying system actually works, because most of the interesting behaviour (and most of the failure modes) only makes sense once you understand the machinery.

---

## What is FrameShift?

FrameShift is a Cursor-meets-Canva-style AI video editor.

The idea is that a user can select an object in a single video frame, make an edit to it, and then propagate that change through the remainder of the video.

Under the hood, the system combines five components, each solving a fundamentally different problem:

- **SAM 2** — a promptable segmentation transformer that finds an object from a click/box and tracks it frame-to-frame using a learned memory.
- **Gemini** — a generative image model that synthesizes new pixels from an image + region + text instruction.
- **Mask-based compositing** — a per-pixel alpha blend that stencils generated pixels into the original footage.
- **RIFE** — a neural frame-interpolation network that invents in-between frames from two real ones.
- **FFmpeg** — the decode/encode layer that turns a compressed video file into editable images and back again.

The problem FrameShift is trying to solve is that editing an individual object in a video is traditionally very time-consuming. In conventional editing software, an editor may need to manually mask or rotoscope the object and then keyframe the edit across the whole clip. This gets dramatically harder when the object moves, changes shape, rotates, or becomes partially occluded — because every one of those changes has to be tracked by hand.

---

## System Pipeline

At a high level, the project divides into five stages: **upload, extraction, segmentation, editing, and rendering.**

### 1. Upload and Temporary Storage

When a user uploads a video, the system stores three main types of data in temporary storage: the original uploaded video, the decoded video frames, and the segmentation masks generated for those frames.

After the initial extraction, most edits operate directly on the decoded frames rather than repeatedly modifying the compressed video file. This matters more than it sounds. A compressed video is *not* a stack of independent images — as you'll see below, most frames are stored as differences from other frames, so you can't just "edit frame 200" in place. By decoding to a flat sequence of images first, every frame becomes an ordinary standalone picture you can paint on.

One subtlety worth noting: the **frames are stored as JPEG (lossy)** while the **masks are stored as PNG (lossless)**. That's deliberate. A mask has to be pixel-exact — a value is either "object" or "not object" — and JPEG's block-based compression would fuzz the edges. JPEG is fine for the frames themselves because they're photographic and a little compression loss is invisible, but you pay for it if you decode → JPEG → re-edit → re-JPEG many times, since each round of lossy compression compounds.

### 2. Video Extraction and Encoding

FFmpeg is used to decode the uploaded video into a sequential set of JPEG frames. A 30-frames-per-second video becomes roughly 30 individual images for every second of footage, each numbered so the frames and their masks stay aligned. The system also supports GPU-accelerated encode/decode when the hardware is available.

To understand why any of this is nontrivial, it's worth looking at what a "video file" actually is.

#### How video decoding works, from the bottom up

**A file is not a codec.** The `.mp4` / `.mov` / `.mkv` you upload is a *container*. A container is just a wrapper that interleaves one or more elementary streams (video, audio, subtitles) along with timestamps and an index telling the player when each chunk should be shown. The container says nothing about how the pixels are compressed. Inside it, the video stream is compressed by a *codec* — H.264, H.265/HEVC, VP9, AV1, etc. FFmpeg first *demuxes* (pulls the streams apart), then *decodes* the video stream. This split is why the same `.mp4` extension can hold wildly different content.

**Why compression is unavoidable.** Raw video is enormous. A single 1920×1080 frame at 24-bit colour is about 6 MB. At 30 fps that's ~186 MB *per second*, or ~11 GB per minute. Codecs routinely hit 100–1000× compression, and they do it by attacking two kinds of redundancy:

- **Spatial redundancy** — neighbouring pixels within one frame are usually similar (a patch of sky is nearly one colour). Handled *inside* a frame.
- **Temporal redundancy** — consecutive frames are nearly identical (only a few things moved). Handled *between* frames.

**Frame types.** This is where the "you can't just edit frame 200" problem comes from. A compressed stream is a sequence of three frame types:

```text
GOP (Group of Pictures):  I  B  B  P  B  B  P  B  B  P ...
```

- **I-frame (intra-coded):** compressed entirely on its own, essentially like a JPEG. A true keyframe and a random-access point.
- **P-frame (predicted):** doesn't store a full picture. Instead the encoder searches the previous frame for a block that matches the current one, stores just the **motion vector** (how far the block moved) plus a small **residual** (what's left over after the prediction). A P-frame can be a tiny fraction of an I-frame's size.
- **B-frame (bi-directional):** predicts from both a past *and* a future frame.

So most frames only make sense relative to their neighbours. To get an independent editable image at position N, the decoder has to walk from the nearest preceding I-frame forward, applying every motion-compensated prediction in between. That's exactly the work FFmpeg does, and exactly why FrameShift decodes everything to flat JPEGs up front.

**The intra-frame pipeline (how a single I-frame or JPEG is actually compressed).** Both JPEG and the intra path of a video codec run roughly this pipeline on small blocks of the image:

1. **Colour transform: RGB → YCbCr.** Split the image into one *luma* (brightness) channel and two *chroma* (colour) channels. The eye is far more sensitive to brightness than to colour, so separating them lets the encoder spend bits where they're noticed.
2. **Chroma subsampling (4:2:0).** Store the chroma channels at a quarter of the samples (half resolution horizontally and vertically). This alone cuts data roughly in half — from 3 values per pixel to an average of 1.5 — with almost no perceptual loss.
3. **Block split + DCT.** The image is chopped into blocks (8×8 in JPEG; variable-size blocks in H.264/265) and each block is run through a **Discrete Cosine Transform**. The DCT rewrites the block from pixel values (spatial domain) into a set of *frequency* coefficients: low frequencies describe smooth gradients, high frequencies describe fine edges and texture. The key property is **energy compaction** — for natural images, almost all of the block's signal lands in a handful of low-frequency coefficients, and most of the high-frequency ones are near zero. (Video codecs technically use an integer approximation of the DCT for speed and exact reversibility, but the idea is the same.)
4. **Quantization.** Each coefficient is divided by a value from a quantization table and rounded. **This is the step where information is actually thrown away** — and it's tuned so that high-frequency coefficients (which the eye barely registers) get crushed hard, often straight to zero. A single "quality" or QP knob scales this table to trade file size against fidelity.
5. **Entropy coding.** After quantization the block is mostly zeros, which run-length encode beautifully. A final lossless pass (Huffman/arithmetic in JPEG, CABAC/CAVLC in H.264) assigns shorter bit-codes to more common symbols.

**Decoding just runs this backwards:** entropy-decode → dequantize → inverse DCT to rebuild each block, and for P/B-frames, add the motion-compensated prediction pulled from previously decoded reference frames held in a buffer.

**GPU acceleration** isn't the shader cores doing the same math faster. Modern GPUs carry *dedicated fixed-function silicon* for video — NVIDIA's NVDEC/NVENC, Intel Quick Sync, AMD VCN, exposed through APIs like VAAPI, VideoToolbox, and DXVA. These are small ASIC blocks that implement entropy decode, inverse transform, and motion compensation directly in hardware. Because the operation is fixed and deeply pipelined, they're vastly more throughput- and power-efficient than a general CPU grinding through the same steps. The tradeoff is a little less flexibility and sometimes slightly worse compression efficiency than a top software encoder like x265 on a slow preset. For FrameShift, offloading decode of the upload and encode of the final render to these blocks is where most of the wall-clock speedup comes from.

**After segmentation**, the system holds two aligned sequences:

```text
frames/                 masks/
  frame_0001.jpg          mask_0001.png
  frame_0002.jpg          mask_0002.png
  frame_0003.jpg          mask_0003.png
```

Each frame has a corresponding mask identifying the selected object.

---

## Editing Pipeline

The user sends a list of requested changes to the `/edit` endpoint (with propagation handled by `/edit/propagate` and AI edits by `/ai/edit/*`). These edits follow one of two general paths: edits that require generative AI, and edits that don't.

---

# Path One: Generative Edits Using Gemini

The Gemini path handles semantic operations — replacing an object, transforming its appearance, or generating something new from a text prompt. For example, the user might select a car and ask the system to turn it into a futuristic vehicle.

Gemini receives an anchor frame, the selected region, and the user's text instruction, and generates edited *keyframes* at intervals — every 15th frame in the current implementation — rather than generating every single frame of the video.

#### How a generative image model fundamentally works

A key thing to hold onto: Gemini is producing an **independent sample** each time it's called. To see why that causes trouble later, it helps to know how these models generate.

Modern image generation and editing is dominated by **diffusion models**. The training idea is almost paradoxically simple: take real images, progressively add random Gaussian noise until they're pure static, and train a large neural network to predict and remove that noise one small step at a time. At generation time you start from pure noise and run the network for many steps, each step nudging the image toward something coherent. To *steer* it, the text instruction is encoded into embeddings and injected via **cross-attention**, so at every denoising step the network is "looking at" the prompt; for image *editing*, the input frame and the selected region are fed in as additional conditioning so the model preserves context outside the edit and only reinvents inside it. (Gemini's exact internals aren't public and it may combine diffusion with autoregressive token prediction, but the conditioning-plus-iterative-refinement picture is the relevant one here.)

The consequence that matters for video: **each call is a fresh stochastic draw.** Two calls on two frames — even with near-identical inputs — sample different noise and land on subtly different results. The model has no memory of what it produced for the previous frame, and no notion that these frames are supposed to depict the *same object a fraction of a second apart*. That is the root of the temporal-consistency problem downstream.

### SAM 2 Mask Propagation

SAM 2 tracks the selected object across the video. In this implementation, masks may be evaluated or corrected at intervals, and SAM 2 uses temporal information to propagate the object mask across the surrounding frames.

#### How SAM 2 actually tracks

SAM 2 (Segment Anything Model 2) is built from a few distinct pieces:

- An **image encoder** — a hierarchical vision transformer (Hiera) that runs once per frame and turns it into a grid of feature embeddings. A vision transformer splits the image into patches and uses self-attention so every patch can incorporate context from every other patch, which is what lets it reason about whole objects rather than isolated pixels.
- A **prompt encoder** that turns the user's input (a point, box, or mask) into embeddings.
- A lightweight **mask decoder** that fuses image features with the prompt and emits a mask plus a confidence score.

The video-tracking magic is a **memory mechanism**: a memory encoder, a memory bank, and memory attention. After SAM 2 segments a frame, it encodes that result into memory. On the next frame, memory attention lets the current frame's features attend back over the stored memories of earlier frames (and the original prompt). In effect the model conditions each new prediction on *what the object looked like and roughly where it was*. It's a spatially-aware, attention-based form of recurrence.

Crucially, SAM 2 tracks by **appearance and spatial memory — not semantics**. It has no idea that Gemini swapped a car for a spaceship; it just keeps chasing the original visual signature. This works well when the object stays visually consistent, and it struggles when:

- the object becomes occluded,
- the lighting shifts significantly,
- the object changes shape,
- the camera moves quickly,
- the edited object looks very different from the original, or
- small errors accumulate and the model gradually **drifts** onto the wrong region.

Most of these are the same failure: the memory it's matching against no longer resembles what's on screen.

### Combining Gemini and SAM 2

The most important part of the pipeline is the integration between Gemini's generated image and SAM 2's mask. The mask acts like a **stencil**: keep Gemini's pixels inside the object region, keep the original video's pixels everywhere else.

#### How the compositing actually works

This is **alpha compositing** — the same "over" operation every image editor uses for layers. Per pixel:

```text
final = Gemini · mask + original · (1 − mask)
```

Where `mask` is 1 inside the object and 0 outside. If the mask is strictly binary you get hard, aliased edges — a visible cut-out look. In practice you usually want a **soft (feathered) mask**, where the value ramps from 1 to 0 over a couple of pixels at the boundary, so the edit blends into the background. Because the masks are stored losslessly (PNG), those alpha values stay exact through the pipeline. The whole point of this operation is to inject Gemini's semantic edit *without* disturbing the rest of the frame — you're not replacing the picture, you're replacing a cut-out region of it.

---

## RIFE Frame Interpolation

Because Gemini only generates selected keyframes, the system needs to fill the temporal gaps. RIFE interpolates the missing frames: given two frames, it predicts the intermediate ones to create smoother motion.

#### How RIFE fundamentally works

RIFE stands for **Real-Time Intermediate Flow Estimation**. The classical way to interpolate a frame at time *t* between frame 0 and frame 1 would be to estimate the optical flow (pixel motion) between the two frames and warp along it — but that gives you flow *between the inputs*, when what you actually need is flow *from the missing middle frame back to each input*, which you don't have.

RIFE's trick is a network (IFNet) that **directly estimates those intermediate flows** in a coarse-to-fine pass — it predicts, for the frame you're trying to synthesize, how to reach into both neighbours. It then **backward-warps** both input frames toward time *t* using those flows and passes the two warped candidates to a small fusion network that blends them and patches up disoccluded areas. Because it's a single feed-forward pass with no iterative optimization, it runs in real time. During training a "teacher" that's allowed to see the true middle frame supervises the flow estimate (privileged distillation).

### Why Morphing Occurs

RIFE assumes its two endpoints are two glimpses of the *same scene* caught mid-motion. That assumption is exactly what breaks here.

When Gemini generates two keyframes **independently**, the object can differ in shape, colour, or detail between them — and there is no real physical motion connecting the two versions, because one wasn't produced by moving the other. RIFE doesn't know this. It dutifully invents a flow field that morphs appearance A into appearance B, and the result **smears, ghosts, stretches, changes shape, or briefly shows features of both versions at once.**

The fundamental mismatch:

- **Gemini** understands the semantic instruction, but is **not temporally consistent** — every keyframe is a fresh draw.
- **RIFE** produces temporally smooth transitions, but is **semantically unaware** — it interpolates pixels, not objects.

SAM 2 provides an object mask, but the Gemini and RIFE stages aren't fully mask-aware *during* generation and interpolation, so SAM 2 alone can't prevent the morphing. Mask compositing can restrict *where* the interpolated result appears, but it can't guarantee the pixels *inside* the mask represent a stable object.

---

## Potential Improvement: Optical Flow

One promising improvement is to use optical flow more directly.

#### What optical flow actually is

Optical flow is the apparent motion of brightness patterns between two images — a vector `(u, v)` for every pixel saying where it went. It rests on the **brightness constancy assumption**: a point keeps the same intensity as it moves, `I(x, y, t) = I(x+dx, y+dy, t+dt)`. Taylor-expanding that gives the **optical flow constraint equation**:

```text
Ix·u + Iy·v + It = 0
```

One equation, two unknowns per pixel — so it's underdetermined. This is the famous **aperture problem**: peering through a small window at a moving edge, you can only tell how it moved *perpendicular* to itself, not along it. Every flow method resolves this with an extra assumption:

- **Lucas–Kanade** assumes flow is constant within a small window and solves a tiny least-squares system — good for sparse, well-textured points.
- **Horn–Schunck** adds a global smoothness constraint and solves for a dense field everywhere.
- **Modern learned methods** (PWC-Net, **RAFT**) build a *cost volume* correlating features between the two frames and iteratively refine a dense flow with a recurrent update network — far more robust to large motion and texture-poor regions.

Once you have flow you can **warp**: move each edited pixel to where the scene actually moved.

#### How this would help FrameShift

Instead of asking Gemini to independently regenerate the object at several points, the system could:

1. Generate the edited object **once**.
2. Estimate the object's real motion from the *original* footage using optical flow.
3. **Warp those exact edited pixels forward** through time along that motion.
4. Use SAM 2 masks to constrain and clean up the propagated result.
5. Regenerate only when the warp becomes unreliable.

The win is temporal consistency: the object's appearance is *carried forward from one real edited frame* rather than re-hallucinated each time, so it can't morph between two different drafts of itself. This wasn't part of the original design, so it wasn't implemented. And it's no silver bullet — flow still fails under occlusion, large perspective changes, or newly revealed surfaces (there's no "source" pixel to warp for something that was hidden) — but it would meaningfully cut the morphing between keyframes.

---

# Path Two: Non-Generative Edits

The second path handles operations that don't need Gemini at all: deleting, blurring, recolouring, filtering, or resizing an object within its existing region.

Take deletion. The user selects an object and presses delete; SAM 2 tracks it through the following frames and produces the masks. Each masked region is then removed and filled using OpenCV's **TELEA inpainting**.

#### What TELEA actually is, and how it works

TELEA is the algorithm from Alexandru Telea's 2004 paper *"An Image Inpainting Technique Based on the Fast Marching Method"* — it's the `cv2.INPAINT_TELEA` option in OpenCV. Its job is to fill a hole (the masked region) using only the surrounding known pixels, and it does so by working **from the boundary of the hole inward**.

The processing *order* is set by the **Fast Marching Method (FMM)**. FMM solves the Eikonal equation `|∇T| = 1`, which computes `T`, the distance of each hole pixel from the boundary. Intuitively it models a front sweeping inward from the edge at constant speed, and — much like Dijkstra's algorithm — it always fills the *closest* not-yet-filled pixel next, using a priority queue. This ordering guarantees that by the time any pixel is filled, its neighbours are already known.

To compute a pixel's actual value, TELEA takes a small neighbourhood of already-known pixels around it and forms a **weighted average — but not a plain one.** For each known neighbour `q`, it estimates `p`'s value as:

```text
I(p) ≈ Σ_q  w(p,q) · [ I(q) + ∇I(q)·(p − q) ]  /  Σ_q w(p,q)
```

The term `I(q) + ∇I(q)·(p − q)` is the important bit: it doesn't just copy `q`'s colour, it *linearly extrapolates along `q`'s gradient*, so smooth intensity ramps and edges continue a short way into the hole instead of flattening out. The weight `w(p, q)` combines three factors: a **directional** term (favour neighbours lined up with the fill direction), a **geometric distance** term (nearer neighbours count more), and a **level-set** term from `T` (neighbours on the same distance-contour count more, keeping the fill consistent along the advancing front).

The net effect is a fast, smooth propagation of colour and structure inward. That's also its limit: it's fundamentally a *diffusion-like interpolation*, so it excels at thin or small regions (scratches, small object removal) but **can't hallucinate texture** — over a large hole with a busy background it goes blurry and smeary. (OpenCV's other option, `INPAINT_NS`, uses a Navier–Stokes fluid-dynamics analogy to propagate isophotes instead, with similar limits.)

Each edited frame is overwritten with its inpainted version before re-encoding.

### Limitation of Frame-by-Frame Inpainting

OpenCV inpainting processes each frame **independently**. It has no temporal memory — it doesn't know what the filled region looked like one frame earlier or later. So the reconstructed background can shift slightly from frame to frame, producing flickering, texture inconsistencies, unstable edges, and visible artifacts wherever the removed object revealed a complex background. This works best when the background is simple or relatively static. (This is precisely the gap a flow-based or dedicated *video* inpainting method would close — by borrowing consistent background from neighbouring frames instead of guessing fresh each time.)

---

## Why Some Edits Work Better Than Others

The architecture is strongest when an edit stays **inside the original object's segmented region** — because the SAM 2 mask that already exists is a perfect stencil for it. That makes it well suited to removing, blurring, recolouring, re-texturing, applying a localized effect, or resizing within roughly the same area.

It's structurally weaker for **moving** or **completely replacing** an object. Moving forces the system to solve two separate problems at once: generate pixels in the object's *new* location, and reconstruct the background that was *hidden behind it* in the old location — and the moved object may spill outside the original mask, so the stencil no longer describes the edited content. Replacing is similar: a new object can have a completely different shape and silhouette, so it may need an entirely new segmentation mask rather than the original object's boundary.

---

## Overall Design Trade-Off

The project demonstrates that object-level video editing is not just an image-generation problem. A working system has to coordinate semantic understanding, object segmentation, motion estimation, temporal consistency, occlusion handling, background reconstruction, and video encoding — all at once.

FrameShift combines several strong models, but each solves a *different* slice of that list:

- **SAM 2** knows *where* the original object is.
- **Gemini** knows *how* to generate the requested transformation.
- **RIFE** knows how to create smooth intermediate frames.
- **OpenCV/TELEA** knows how to fill missing image regions.

None of them individually maintains a complete, persistent understanding of the *edited* object across time — and every seam in the pipeline is a place where one model's assumptions meet another's and disagree (Gemini's stochastic draws vs. RIFE's motion assumption; SAM 2's appearance-tracking vs. a newly-generated object it's never seen; per-frame inpainting vs. a background that should stay stable).

The strongest result, then, isn't just the editor itself — it's the exploration of how these systems can be wired together, and exactly where their assumptions begin to conflict.
