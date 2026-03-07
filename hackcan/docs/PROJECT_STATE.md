# FrameShift AI — Project State

**Last updated:** 2026-03-07
**Branch:** main
**Status:** Backend complete. /edit is now async with progress tracking. Frontend landing + editor built. Critical editor bug: edit request format mismatch. SAM 2 added to requirements.txt but not yet tested.

---

## What Is This

FrameShift AI is an AI-powered video editor. The core idea: upload a short video, click an object in any frame, apply an edit (recolor, resize, replace, add, delete), and the edit propagates across every frame automatically.

It uses:
- **YOLOv11** for object detection
- **SAM 2** for segmentation + mask propagation
- **Cloudinary** for per-frame image transforms and final video delivery
- **FFmpeg** for frame extraction and re-encoding

---

## Repository Layout

```
hackcan/
├── backend/
│   ├── main.py                      # FastAPI app — all endpoints
│   ├── requirements.txt             # Python deps
│   ├── .env                         # Cloudinary credentials (not committed)
│   ├── services/
│   │   ├── cloudinary_service.py    # Upload, transform, download helpers
│   │   ├── ffmpeg_service.py        # Frame extraction + video encoding
│   │   ├── project_manager.py       # Project dir + status.json management
│   │   ├── sam2_service.py          # SAM 2 segmentation + mask propagation
│   │   └── yolo_service.py          # YOLOv11 detection
│   ├── scripts/                     # One-off dev scripts (extract_frame, yolo_detect, etc.)
│   └── input/                       # Sample test video
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx                 # Landing page
│   │   └── editor/[projectId]/
│   │       └── page.tsx             # Editor page
│   ├── src/components/
│   │   ├── DropZone.tsx             # Drag-and-drop upload → /upload → /extract → redirect
│   │   ├── Hero.tsx, TopBar.tsx, Footer.tsx, FeatureCarousel.tsx, etc.
│   └── src/lib/
│       ├── cloudinary.ts            # Cloudinary URL-gen client
│       └── media.ts                 # getVideo, getVideoThumbnail helpers
└── docs/plans/
    ├── 2026-03-07-frameshift-pipeline-design.md    # Original pipeline design
    ├── 2026-03-07-frameshift-implementation.md     # Original 9-task implementation plan
    └── 2026-03-07-backend-completeness.md          # Second plan (add/delete edits, background tasks)
```

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind v4, Konva.js |
| Backend | Python, FastAPI, Uvicorn |
| AI — Detection | YOLOv11 (ultralytics) |
| AI — Segmentation | SAM 2 (Meta) — `sam2==1.1.0` in requirements.txt |
| Video Processing | FFmpeg |
| Media Platform | Cloudinary (upload, generative AI transforms, CDN) |
| Font | Clash Display (variable woff2) |

---

## Backend — API Endpoints

All endpoints live in `hackcan/backend/main.py`.

### `GET /health`
Returns `{ status: "ok" }`.

---

### `POST /upload`
Accepts a video file (multipart form). Creates a new project directory, saves the file locally, uploads it to Cloudinary, and returns:
```json
{ "project_id": "abc12345", "video_url": "https://...", "public_id": "frameshift/..." }
```
Project data lives in `/tmp/frameshift/{project_id}/`.

---

### `POST /extract`
Input: `{ project_id }`.
Kicks off a **BackgroundTask** that:
1. Extracts frames at 30fps via FFmpeg → `/tmp/frameshift/{id}/frames/frame_0001.jpg ...`
2. Runs YOLOv11 on every frame, caches detections in `status.json`
3. Updates project status: `created → processing → extracting → detecting → ready`

Returns immediately with `{ status: "processing" }`. Frontend polls `/project/{id}/status`.

---

### `GET /project/{project_id}/status`
Returns the contents of the project's `status.json`:
```json
{
  "status": "ready",
  "frame_count": 150,
  "detections": {
    "1": [{ "label": "person", "confidence": 0.91, "bbox": [x1,y1,x2,y2] }],
    ...
  }
}
```
Status progression: `created → processing → extracting → detecting → ready`

---

### `GET /frame/{project_id}/{frame_index}`
Serves a single extracted frame as `image/jpeg`. Used by the editor canvas.

---

### `POST /detect`
Input: `{ project_id, frame_index }`.
Runs YOLOv11 on a specific frame on demand.
Returns: `{ objects: [{ label, confidence, bbox }] }`

---

### `POST /segment`
Input: `{ project_id, frame_index, click_x, click_y }`.
Runs SAM 2 on the anchor frame at the clicked point, then propagates the mask across all frames.
Returns: `{ mask_count, anchor_frame }`

> **Note:** Currently uses a simplified propagation — copies the anchor mask to all frames. Full SAM 2 `propagate_in_video()` is a stretch goal.

---

### `GET /mask/{project_id}/{mask_index}`
Serves a single mask PNG.

---

### `POST /edit`
Accepts a list of edit rules and immediately kicks off a **BackgroundTask**. Returns `{ edit_status: "uploading" }`. Poll `/project/{id}/status` for `edit_status` and `edit_progress`.

**`edit_status` progression:** `uploading → editing → done` (or `error`)

**`edit_progress`:** `{ "done": N, "total": M }` — updated after each frame is processed.

Accepts a list of edit rules, each with a frame range:

```json
{
  "project_id": "abc12345",
  "edit_rules": [
    {
      "edit_type": "recolor",
      "start_frame": 1,
      "end_frame": 150,
      "color": "FF0000"
    }
  ]
}
```

**Supported `edit_type` values:**

| Type | What it does | Cloudinary mechanism |
|------|-------------|----------------------|
| `recolor` | Colorize the masked region | `e_colorize` overlay |
| `resize` | Scale the object up or down | `c_scale` overlay |
| `replace` | Swap the object with another asset | Overlay at mask bbox |
| `delete` | Erase the object using generative AI | `e_gen_remove` |
| `add` | Generate a new object from a text prompt | `gen_replace` |

Processes 20 frames concurrently. Frames outside any rule's range are copied as-is.

---

### `POST /render`
Input: `{ project_id }`.
Guarded: returns an error if `edit_status` is not `done`. FFmpeg re-encodes all edited frames into `output.mp4`, uploads to Cloudinary, returns:
```json
{ "video_url": "https://...", "cloudinary_public_id": "..." }
```

---

## Cloudinary Services (`cloudinary_service.py`)

| Function | Purpose |
|----------|---------|
| `configure()` | Reads env vars, initialises Cloudinary SDK |
| `upload_file(path, folder, resource_type)` | Uploads image or video, returns `public_id` + `url` |
| `get_url(public_id, transformations)` | Builds a Cloudinary URL with optional transforms |
| `apply_recolor(frame_id, mask_id, color)` | Colorize via `e_colorize:80` overlay |
| `apply_resize(frame_id, mask_id, x,y,w,h, scale)` | Scale object and re-overlay |
| `apply_replace(frame_id, mask_id, x,y,w,h)` | Overlay replacement asset at bbox |
| `apply_delete(frame_id, mask_id)` | Generative remove via `e_gen_remove` |
| `apply_add(frame_id, prompt, x,y,w,h)` | Generative replace via `gen_replace` |
| `download_url(url, save_path)` | Async download of transformed frame |

---

## Frontend

### Landing page (`src/app/page.tsx`)
Full marketing page with animated sections: Hero, FeatureCarousel, HowItWorks, BentoGrid, StickyCTA, Footer. Uses Clash Display variable font. The `DropZone` component is embedded.

### DropZone (`src/components/DropZone.tsx`)
- Accepts drag-and-drop or file picker (video files only)
- Calls `POST /upload` → `POST /extract` → immediately redirects to `/editor/{project_id}`
- Shows spinner + status text ("Uploading video..." / "Extracting frames...") during upload
- **Does NOT poll** `/project/{id}/status` before redirecting — the editor will load before frames are ready

### Editor page (`src/app/editor/[projectId]/page.tsx`)
- On mount: fetches `/project/{id}/status` **once** (not a polling loop) to get `frame_count`; defaults to 300 while waiting
- Loads frame 1 and calls `POST /detect` to show YOLO bboxes
- **Konva.js canvas**: renders the current frame + YOLO bboxes as red rectangles
- **Frame scrubber**: range slider 1 → `frame_count`, calls `POST /detect` per frame on change
- **Click to segment**: canvas click → `POST /segment` with scaled pixel coordinates
- **Edit panel dropdown**: only exposes `recolor`, `resize`, `replace` (delete and add not in UI)
- **Apply Edit flow**: `POST /edit` (broken format — see bugs) → `POST /render` → shows result video

---

## Environment Variables

### Backend (`hackcan/backend/.env`)
```
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
```

### Frontend (`hackcan/frontend/.env.local`)
```
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=...
NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET=...
NEXT_PUBLIC_API_URL=http://localhost:8000   ← not yet set, defaults in code
```

---

## What Works

- [x] Full backend pipeline: upload → extract (background) → detect → segment → edit (5 types) → render
- [x] `/project/{id}/status` polling endpoint with frame count + cached detections
- [x] All 5 Cloudinary edit types implemented in the backend (recolor, resize, replace, delete, add)
- [x] Batch edit rules with frame ranges in backend
- [x] `/edit` runs as a background task — returns immediately, progress tracked via `edit_status` / `edit_progress` in `status.json`
- [x] `/render` guarded: returns error if `edit_status` is not `done`
- [x] Frontend landing page with animations and DropZone
- [x] Editor page: Konva canvas, YOLO bboxes, frame scrubber, segment on click, result video display
- [x] `requirements.txt` includes all backend dependencies including `sam2==1.1.0`

## What Doesn't Work / Needs Fixing

### Critical bugs

- [ ] **Edit request format mismatch** — `editor/page.tsx:handleApplyEdit` sends:
  ```json
  { "project_id": "...", "edit_type": "recolor", "color": "FF0000" }
  ```
  But the backend `/edit` expects:
  ```json
  { "project_id": "...", "edit_rules": [{ "edit_type": "recolor", "start_frame": 1, "end_frame": 150, "color": "FF0000" }] }
  ```
  The edit will silently fail (Pydantic will reject the body, returning a 422). Frontend never checks the response from `/edit`.

### Missing features / UX gaps

- [ ] **SAM 2 not end-to-end tested** — `sam2==1.1.0` is now in `requirements.txt` (pip-installable). Still need to verify the import works, download the checkpoint, and confirm `/segment` doesn't crash.
- [ ] **No status polling in DropZone** — redirects to editor immediately after `/extract` returns; editor loads before frames/detections are ready. Need a polling loop on `/project/{id}/status` in the editor (or in DropZone before redirect) that waits for `status === "ready"`.
- [ ] **Editor uses on-demand `/detect` instead of cached detections** — the background task pre-computes all detections into `status.json`, but the editor calls `POST /detect` per frame instead of reading from the already-fetched status data. Causes redundant YOLO inference on every frame scrub.
- [ ] **Delete and Add edit types not in UI** — backend supports them but the editor dropdown only shows recolor, resize, replace. No UI for the prompt input (add) or confirmation (delete).
- [ ] **Replace edit type has no asset input** — the option appears in the dropdown but there is no field to supply the replacement asset's `public_id`.
- [ ] **`NEXT_PUBLIC_API_URL` not set** in `.env.local` — fine for local dev (hardcoded default `localhost:8000`), must be configured for any deployment.

### Never tested

- [ ] **End-to-end integration** — the full upload → extract → segment → edit → render flow has not been run against the live backend.

---

## Known Issues / Gotchas

- `apply_delete` uses `e_gen_remove` which requires a Cloudinary plan that supports generative AI features
- `apply_add` uses `gen_replace` syntax — needs verification against Cloudinary docs, the URL format may need adjustment
- SAM 2 propagation is currently naive (copies anchor mask to all frames) — real tracking via `propagate_in_video()` is a stretch goal
- CORS is locked to `http://localhost:3000` in `main.py` — must be updated for any deployment

---

## Priority Order for Next Work

1. **Fix edit request format** in `editor/page.tsx:handleApplyEdit` — add `edit_rules` wrapper, `start_frame`/`end_frame` (default to 1 / frameCount); poll `/project/{id}/status` until `edit_status === "done"` before calling `/render`
2. **Add status polling** in the editor — poll `/project/{id}/status` on mount until `status === "ready"`, show loading state, then enable canvas
3. **Use cached detections** — after polling resolves, read detections from the status response instead of calling `/detect` per frame
4. **Install SAM 2** — manual install from Meta GitHub + checkpoint download (see How to Run)
5. **Add delete/add UI** — add to dropdown, prompt input for add, confirmation for delete
6. **Fix replace UI** — add a Cloudinary public_id input for the replacement asset
7. **End-to-end test** — run the full pipeline on a real video

---

## How to Run Locally

### Backend
```bash
cd hackcan/backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Create .env with Cloudinary credentials
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd hackcan/frontend
npm install
# Create .env.local with NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME and NEXT_PUBLIC_API_URL
npm run dev
# Visit http://localhost:3000
```

### SAM 2
SAM 2 is now in `requirements.txt` as `sam2==1.1.0` and installs via pip with the rest of the dependencies. Still need to download the model checkpoint manually:
```bash
mkdir -p hackcan/backend/checkpoints
# Download sam2_hiera_large.pt to hackcan/backend/checkpoints/
# https://github.com/facebookresearch/sam2#model-checkpoints
```
