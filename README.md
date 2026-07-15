<p align="center">
  <img src="docs/Thumbnail.png" alt="FrameShift Logo" width="400">
</p>

<p align="center">
  <strong>AI-powered video editing. Click an object, apply an edit, propagate across every frame.</strong>
</p>

## Overview

FrameShift lets you edit objects in video without frame-by-frame manual work. Upload a video, click any object, and apply edits like recolor, remove, resize, or replace — the changes automatically propagate across your entire clip.

<p align="center">
  <img src="docs/Image_1.png" alt="Landing Page" width="100%">
</p>

## Features

<p align="center">
  <img src="docs/Image_2.png" alt="Feature Overview" width="100%">
</p>

- **AI Object Detection** — YOLOv11 automatically detects every object in your video. Select any object just by clicking on it.
- **Background Removal** — Remove or swap backgrounds instantly with AI-generated scenes.
- **Color Grading** — Change the color of any detected object across all frames.
- **Smart Resize** — Reformat for any platform in one click.

## How It Works

### 1. Upload your video
Drag and drop or upload from your device. FrameShift extracts every frame and runs object detection in the background.

### 2. Select an object
Click any detected object in the editor. SAM 2 segments it precisely and tracks it across frames.

<p align="center">
  <img src="docs/Image_4.png" alt="Object Selection" width="100%">
</p>

### 3. Apply edits
Use the toolbar to remove, recolor, resize, blur, or replace objects. Apply whole-frame effects like background removal, enhance, upscale, or restore. Or describe your edit in natural language with the AI Edit panel.

<p align="center">
  <img src="docs/Image_3.png" alt="Editor View" width="100%">
</p>

### 4. Propagate to all frames
One click propagates your edit across the entire frame range. Preview the result, then render your final video.

<p align="center">
  <img src="docs/Image_6.png" alt="AI Edit Applied" width="100%">
</p>

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js, Tailwind CSS, Auth0 |
| Backend | FastAPI, Python |
| Object Detection | YOLOv11 (Ultralytics) |
| Segmentation & Tracking | SAM 2 |
| Generative Edits | Gemini (keyframes) |
| Frame Interpolation | RIFE (FILM optional fallback) |
| Inpainting / Local Edits | OpenCV (TELEA) |
| Storage & Media | Mounted project volume + local filesystem |
| Database | Supabase (Postgres) |
| Frame Processing | FFmpeg |

## Architecture

```
Video Upload → Project Volume → FFmpeg Frame Extraction → YOLOv11 Detection
                                              ↓
              Click Object → SAM 2 Segmentation + Mask Propagation
                                              ↓
        ┌─ Generative edits: Gemini keyframes (every 15th frame)
        │       → mask-based alpha compositing → RIFE interpolates the gaps
        └─ Local edits: OpenCV per frame (TELEA inpainting, recolor, blur, resize)
                                              ↓
              FFmpeg Re-encode → Final Video on the project volume
```

For a deep dive into how each stage works (video codecs, SAM 2's memory mechanism, diffusion models, RIFE, TELEA, and where their assumptions conflict), see [docs/Technical-Overview.md](docs/Technical-Overview.md).

## Supported Edits

### Object-Level (SAM 2 mask)

| Edit | Description |
|---|---|
| Recolor | Change the color of a specific object |
| Resize | Scale an object in place |
| Remove | AI-powered object removal |
| Replace | Swap an object with AI-generated content from a text prompt |
| Blur | Blur only the masked region (faces, license plates, etc.) |

### Whole-Frame

| Effect | Description |
|---|---|
| Background Remove | Remove background, keep foreground |
| Background Replace | Replace background with an AI-generated scene |
| Enhance | AI quality and detail improvement |
| Upscale | AI resolution upscaling |
| Restore | Fix compression artifacts and noise |
| Blur | Apply blur to the entire frame |

### AI Edit

Describe your edit in natural language (e.g., *"make the person in the middle red"*) and FrameShift applies it.

<p align="center">
  <img src="docs/Image_5.png" alt="Editor with AI Edit" width="100%">
</p>

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- Durable production volume mounted at the projects path
- Supabase project
- Auth0 application

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the SAM 2 checkpoint:

```bash
mkdir checkpoints
curl -L -o checkpoints/sam2_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

Create `backend/.env`:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
AUTH0_DOMAIN=your_auth0_domain
AUTH0_AUDIENCE=your_auth0_audience
```

Start the server:

```bash
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_PROFILE_ROUTE=/api/auth/profile
AUTH0_SECRET=your_secret
AUTH0_BASE_URL=http://localhost:3000
AUTH0_ISSUER_BASE_URL=https://your-tenant.us.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

Start the dev server:

```bash
npm run dev
```

App runs at [http://localhost:3000](http://localhost:3000).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Create project, store video on the project volume |
| `POST` | `/extract` | Extract frames with FFmpeg, run YOLO detection |
| `GET` | `/project/{id}/status` | Poll extraction status and frame count |
| `GET` | `/frame/{id}/{index}` | Serve a specific frame as JPEG |
| `POST` | `/detect` | On-demand YOLO detection for a single frame |
| `POST` | `/segment` | SAM 2 segmentation + propagation |
| `POST` | `/edit` | Apply batch edit rules across frame ranges |
| `POST` | `/render` | Re-encode video with FFmpeg and store the result on the project volume |
