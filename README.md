<p align="center">
  <img src="docs/Thumbnail.png" alt="FrameShift Logo" width="400">
</p>

<p align="center">
  <strong>Click an object, preview one edit, and carry the approved result through the video.</strong>
</p>

<p align="center">
  <a href="https://frameshift-snowy.vercel.app">Live app</a>
  ·
  <a href="docs/frameshift-production-pipeline.md">Production pipeline</a>
  ·
  <a href="docs/modal-cloud-deployment.md">Deployment guide</a>
</p>

## Overview

FrameShift is an object-aware video editor for short clips. Upload a video, click an object on any keyframe, confirm its SAM 2 mask, and apply edits such as remove, recolor, resize, move, blur, glow, or replace. Every tool shows the keyframe result first and asks for confirmation before changing the selected frame range.

The current product accepts videos under six seconds and up to 50 MB. Guests can try the editor without creating an account. Signing in with Auth0 adds private, account-scoped project history and lets projects resume from media and checkpoints stored in Supabase.

<p align="center">
  <img src="docs/Image_1.png" alt="Landing Page" width="100%">
</p>

## Features

<p align="center">
  <img src="docs/Image_2.png" alt="Feature Overview" width="100%">
</p>

- **Click-to-segment** — a positive point prompt runs the SAM 2 image predictor on the current keyframe. The UI displays the mask outline before any clip-wide work begins.
- **Confirmation-first tracking** — SAM 2 tracks the selected object through every frame only after the user approves the keyframe mask.
- **Preview before propagation** — every tool renders one keyframe first. The user can confirm “Apply to all frames” or cancel and revise the edit.
- **Temporally coherent transport** — edits that reveal or generate pixels use lazy, batched RAFT optical flow, bidirectional warping, validity checks, and flow-guided inpainting.
- **Prompt-guided replacement** — Gemini validates object/background prompts, creates the approved keyframe appearance, and FrameShift transports it instead of generating every frame independently.
- **Stackable edits** — after one propagation finishes, another tool starts from the latest edited result. A frame backup supports undoing the latest edit.
- **Durable projects** — Supabase Postgres stores account-scoped metadata; private Supabase Storage stores the original video, current edit, thumbnail, mask checkpoint, and exported MP4.
- **GPU production backend** — FastAPI, SAM 2, and RAFT run on one Modal NVIDIA GPU. A process-wide queue prevents SAM 2 and RAFT from competing for GPU memory.

## How It Works

### 1. Upload your video

Drag in a video under six seconds. The browser validates its duration, uploads it to private Supabase Storage through a signed URL, and asks the Modal backend to hydrate a temporary processing cache. FFmpeg extracts frames at the source frame rate and a 720p working resolution.

### 2. Select an object

Move to any keyframe and click the object. SAM 2 segments only that frame first, and FrameShift draws the red mask outline. Confirm the selection to run SAM 2 video tracking forward and backward across the clip.

<p align="center">
  <img src="docs/Image_4.png" alt="Object Selection" width="100%">
</p>

### 3. Apply an edit

Choose Remove, Recolor, Resize, Blur, Move, Color Pop, Glow, Replace, or Replace BG. FrameShift creates a single-frame preview without mutating the project. Replace tools first validate the prompt and then generate a Gemini keyframe preview.

<p align="center">
  <img src="docs/Image_3.png" alt="Editor View" width="100%">
</p>

### 4. Propagate and export

Confirm “Apply to all frames” to process the selected timeline range. Deterministic tools run directly against each stabilized mask. Motion-dependent tools compute only the RAFT flow pairs they need, cache those flows, and transport or fill pixels through time. The completed edit is synchronized to Supabase, and Export MP4 re-encodes the latest frames with FFmpeg.

<p align="center">
  <img src="docs/Image_6.png" alt="AI Edit Applied" width="100%">
</p>

## Why the Pipeline Moved Away from RIFE Interpolation

The original FrameShift prototype generated edited keyframes at intervals and used RIFE to synthesize the frames between them. RIFE is effective at motion interpolation when both endpoint frames depict the same appearance. It does not solve the harder edit-consistency problem: two independent image-generation calls can change texture, shape, lighting, or identity. Interpolating those mismatched endpoints produces a smooth but visible morph.

The production pipeline now approves one edited anchor and treats it as a layer to transport:

1. RAFT estimates forward and backward correspondence on the original footage.
2. The approved RGB edit and alpha mask are star-warped from the nearest anchor.
3. Forward/backward consistency identifies occlusions and unreliable pixels.
4. Bidirectional layers are blended where both directions are valid.
5. Flow-guided donor frames fill revealed areas; OpenCV TELEA handles only residual holes.
6. If a long chain degrades, refinement starts from the already-warped appearance and the original reference rather than making a fresh, unrelated draw.

This change removes independent generation from the frame loop, reduces morphing, and makes propagation substantially cheaper. The RIFE/FILM modules remain in the repository as legacy experiments, but neither is part of the active production edit path.

## Active Production Architecture

```text
Browser (≤6 s, ≤50 MB)
  → Next.js signed upload route
  → private Supabase Storage: original.mp4
  → signed cache hydration request
  → Modal FastAPI + temporary Modal Volume cache
  → FFmpeg frame extraction (native FPS, 720p working size)

Click keyframe
  → SAM 2 image segmentation
  → user confirms mask
  → SAM 2 video tracking on every frame
  → stabilized masks + Supabase checkpoint

Choose edit
  → one-frame local or Gemini preview
  → user confirms propagation
  ├─ Recolor / Blur / Color Pop / Glow: deterministic OpenCV per frame
  └─ Remove / Move / Resize-down / Replace / Replace BG:
       lazy batched RAFT → warp / composite / flow-guided inpaint

Completed edit
  → FFmpeg current.mp4 → Supabase Storage
  → Export MP4 → Supabase Storage + browser download
```

RAFT is deliberately deferred until an approved edit needs motion. SAM 2 and RAFT are serialized on the same GPU rather than running concurrently. Flow files are cached and reused by later edits.

The default deployed replacement engine is **Backend A**, implemented with RAFT and PyTorch `grid_sample`. The repository also contains an Ezsynth-inspired guided PatchMatch **Backend B** and a CUDA-kernel track behind feature flags. Those paths are testable but are disabled in the default Modal deployment so they cannot block production.

## Tech Stack

| Layer | Active technology |
|---|---|
| Frontend | Next.js 16, React 19, Tailwind CSS 4, Zustand |
| Authentication | Auth0 Next.js SDK; optional guest workflow |
| Account metadata | Supabase Postgres through authenticated Next.js server routes |
| Durable media | Private Supabase Storage with signed upload/download URLs |
| GPU backend | FastAPI on Modal with one selectable NVIDIA GPU |
| Working cache | Modal Volume locally mounted at the project directory |
| Segmentation and tracking | SAM 2 image predictor + video predictor |
| Optical flow | Torchvision RAFT Large, computed lazily in batches |
| Generative previews | Gemini prompt validation and image generation |
| Temporal propagation | PyTorch warping, flow consistency, bidirectional blending |
| Local edits and residual fill | OpenCV, flow-guided donor fill, TELEA fallback |
| Video processing | FFmpeg extraction and H.264 MP4 encoding |

YOLOv11, RIFE/FILM, and guided PatchMatch experiments remain in the source tree. Automatic YOLO detection and RIFE interpolation are disabled in the active pipeline; current object selection is a direct click prompt to SAM 2.

## Supported Edits

### Object-level edits

All object tools use the propagated SAM 2 mask.

| Edit | Current implementation |
|---|---|
| Remove | Flow-guided temporal inpainting with TELEA residual fill |
| Recolor | Deterministic masked color transform |
| Resize | Masked cutout scaling; shrinking uses temporal inpainting for revealed pixels |
| Blur | Gaussian blur limited to the selected object |
| Move | Transports the cutout and fills its vacated region from neighboring frames |
| Color Pop | Keeps the object in color while desaturating the rest of the frame |
| Glow | Adds a feathered bloom around the tracked object |
| Replace | Gemini keyframe appearance propagated with RAFT and validity-aware compositing |

### Whole-frame composition

| Edit | Current implementation |
|---|---|
| Replace BG | Keeps the segmented foreground, generates one background plate, softens the matte, and follows camera motion with scene flow |

### Prompt-guided previews

Natural-language prompts are accepted by Replace and Replace BG. Gemini checks that a prompt belongs to the active tool before image generation. Unrelated, vague, or tool-mismatched requests return a user-facing “Try again with a better prompt” message.

<p align="center">
  <img src="docs/Image_5.png" alt="Editor with AI Edit" width="100%">
</p>

## Project Persistence and Accounts

- Guests can upload and edit without logging in, but do not receive account history.
- Authenticated projects are keyed by the Auth0 `sub` and filtered server-side for every list, update, resume, and delete operation.
- Supabase Postgres stores names, ownership, status, resume frame, frame count, edit version, and media object paths.
- Supabase Storage is the source of truth for `original.mp4`, `current.mp4`, `thumbnail.jpg`, `checkpoint.tar.gz`, and `exports/final.mp4`.
- Modal Volume is a hot processing cache. Opening an older project creates signed Supabase URLs and hydrates the cache through a shared-secret backend route.

## Getting Started

### Prerequisites

- Node.js 20.9 or newer
- Python 3.12 recommended
- [FFmpeg](https://ffmpeg.org/download.html) on `PATH`
- A Supabase project
- An Auth0 Regular Web Application and Auth0 API
- A Gemini API key for Replace and Replace BG
- A Modal account for the production CUDA backend

### 1. Install the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the SAM 2 checkpoint, which is intentionally excluded from Git:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam2_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

Create the local backend configuration:

```bash
cp .env.example .env
```

```dotenv
GEMINI_API_KEY=
FRONTEND_ORIGIN=http://localhost:3000
AUTH0_DOMAIN=
AUTH0_AUDIENCE=https://api.frameshift.app
SUPABASE_URL=
SUPABASE_SECRET_KEY=
SUPABASE_MEDIA_BUCKET=project-media
FRAMESHIFT_IMPORT_SECRET=
FRAMESHIFT_PRECOMPUTE_FLOWS=0
FRAMESHIFT_RAFT_BATCH_SIZE=1
FRAMESHIFT_USE_SYNTH=0
FRAMESHIFT_USE_CUDA_KERNEL=0
```

`FRAMESHIFT_IMPORT_SECRET` is a private shared value used only between the Next.js server and FastAPI cache-hydration route. Use the same value in both environments.

Start FastAPI locally:

```bash
venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Configure Supabase

Apply the SQL migrations in [`supabase/migrations`](supabase/migrations). They create the private `project-media` bucket and the account-scoped `projects` metadata table.

The Supabase secret/service-role key must remain server-only. Never expose it through a variable prefixed with `NEXT_PUBLIC_`.

### 3. Install the frontend

```bash
cd frontend
npm install
cp .env.example .env.local
```

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000

AUTH0_DOMAIN=your-tenant.us.auth0.com
AUTH0_CLIENT_ID=
AUTH0_CLIENT_SECRET=
AUTH0_SECRET=
APP_BASE_URL=http://localhost:3000
AUTH0_AUDIENCE=https://api.frameshift.app

NEXT_PUBLIC_SUPABASE_URL=
SUPABASE_SECRET_KEY=
FRAMESHIFT_IMPORT_SECRET=
```

Generate the Auth0 cookie secret with:

```bash
openssl rand -hex 32
```

Start Next.js:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Production Deployment

### Frontend: Vercel

Import this repository as one Vercel project and set the Root Directory to `frontend`. Keep the detected Next.js build, output, and install commands. Copy the variables from `frontend/.env.example`, set `NEXT_PUBLIC_API_URL` to the Modal web endpoint, and set `APP_BASE_URL` to the canonical Vercel or custom domain.

Add the production domain to the Auth0 application's Allowed Callback URLs, Allowed Logout URLs, and Allowed Web Origins. `NEXT_PUBLIC_` variables are compiled into the browser bundle, so redeploy after changing them.

### Backend: Modal

The backend does not need Render or Railway. Modal supplies both the FastAPI web endpoint and the NVIDIA GPU compute.

```bash
cd backend
cp .env.modal.example .env.modal
venv/bin/modal secret create frameshift-secrets --from-dotenv .env.modal --force
FRAMESHIFT_MODAL_GPU=L4 venv/bin/modal deploy modal_app.py
```

Supported GPU selectors are `L4`, `A10`, `L40S`, and `H100`. `FRONTEND_ORIGIN` in `.env.modal` must exactly match the production frontend origin without a trailing slash. See [`docs/modal-cloud-deployment.md`](docs/modal-cloud-deployment.md) for the full deployment procedure.

## Key API Endpoints

### Next.js server routes

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/uploads/intent` | Validate metadata and create a signed Supabase upload URL |
| `PUT` | `/api/uploads/{projectId}/content` | Same-origin upload fallback |
| `POST` | `/api/uploads/{projectId}/complete` | Mark storage complete and hydrate the Modal cache |
| `GET`, `POST` | `/api/projects` | List or register projects for the current Auth0 user |
| `GET`, `PATCH`, `DELETE` | `/api/projects/{projectId}` | Read, update, or delete an owned project |
| `POST` | `/api/projects/{projectId}/resume` | Restore an owned project into the Modal cache |
| `GET` | `/api/projects/{projectId}/media` | Redirect to a short-lived signed media URL |

### Modal FastAPI routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Report backend, CUDA, and Auth0 status |
| `POST` | `/project/import` | Hydrate the processing cache from signed storage URLs |
| `POST` | `/extract` | Extract native-FPS frames at the working resolution |
| `GET` | `/project/{id}/status` | Poll extraction, segmentation, flow, edit, and storage progress |
| `GET` | `/frame/{id}/{index}` | Serve a working JPEG frame |
| `POST` | `/segment` | Produce only the clicked keyframe mask |
| `POST` | `/segment/propagate` | Track the approved mask across all frames |
| `GET` | `/mask-outline/{id}/{index}` | Serve the transparent mask contour used by the editor |
| `POST` | `/flows/precompute` | Optional manual RAFT precomputation; production normally stays lazy |
| `POST` | `/edit/preview` | Render a non-destructive single-frame edit preview |
| `POST` | `/edit` | Apply confirmed edit rules over a frame range |
| `POST` | `/edit/undo` | Restore the latest frame backup |
| `POST` | `/edit/cancel` | Cancel an active edit job |
| `POST` | `/render` | Encode and persist the final MP4 |
| `GET` | `/render/{id}/video` | Stream the locally cached render |

## Further Reading

- [`docs/frameshift-production-pipeline.md`](docs/frameshift-production-pipeline.md) — temporal consistency, edit-layer propagation, and per-tool design
- [`docs/frameshift-cuda-track.md`](docs/frameshift-cuda-track.md) — custom warp and guided PatchMatch CUDA work
- [`docs/modal-cloud-deployment.md`](docs/modal-cloud-deployment.md) — serialized CUDA deployment and environment setup
- [`docs/auth0-project-history.md`](docs/auth0-project-history.md) — authentication and account-scoped project persistence
- [`docs/Technical-Overview.md`](docs/Technical-Overview.md) — background on video codecs and the original model experiments

## Verification

```bash
# Frontend
cd frontend && npm run build

# Backend
cd backend && venv/bin/pytest -q
```
