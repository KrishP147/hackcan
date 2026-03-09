# FrameShift

By Bryan Lin, Justin Wang, Daniel Zhao, and Krish Punjabi.

AI-powered video editor. Upload a video, click an object, apply edits — changes propagate across every frame automatically. **Cloudinary** is used for image and video storage and for per-frame transforms (including thumbnails).

## What it does

- Upload a video and FrameShift extracts all frames and runs object detection (YOLOv11) across them in the background
- In the editor, click any detected object to segment it with SAM 2
- Apply edits (recolor, resize, delete, replace) — the edit rule runs across a frame range
- Render the final video and get it back via Cloudinary
- Dashboard shows all your projects with thumbnail previews — hover a card and click the image icon to generate a random thumbnail from your video frames, stored and served via Cloudinary

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 15, Tailwind CSS, Auth0 |
| Backend | FastAPI, Python |
| Object detection | YOLOv11 (Ultralytics) |
| Segmentation | SAM 2 |
| Storage & media | Cloudinary (image/video storage, transforms, thumbnails) |
| Database | Supabase (Postgres) |
| Frame processing | FFmpeg |

## Setup

### Prerequisites

- Node.js 18+
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- Cloudinary account
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
curl -L -o checkpoints/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

Create `backend/.env`:

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
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
AUTH0_SECRET=your_secret
AUTH0_BASE_URL=http://localhost:3000
AUTH0_ISSUER_BASE_URL=https://your-tenant.us.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=your_cloud_name
NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET=your_upload_preset
```

Start the dev server:

```bash
npm run dev
```

App runs at `http://localhost:3000`.

## Cloudinary (storage, transforms, thumbnails)

FrameShift uses **Cloudinary** for image and video storage and for per-frame transforms (including thumbnails). Uploaded videos and extracted frames are stored in Cloudinary; the transformation pipeline applies edits via `CloudinaryImage.build_url()`. Thumbnails for the dashboard are generated and served from Cloudinary. All transformation URLs are built in the backend and results are downloaded locally before FFmpeg re-encodes the final video.

### Edit types (object-aware, use SAM 2 mask)

| Edit type | Cloudinary effect | Description |
|---|---|---|
| Recolor | `gen_recolor:prompt_<obj>;to-color_<hex>` | Recolor a specific object by prompt or mask |
| Resize | Overlay crop + scale + `layer_apply` | Crop the object region and scale it in place |
| Delete | `gen_remove` | AI-powered object removal |
| Replace | `gen_replace:from_auto;to_<prompt>` | Swap object with AI-generated content from a text prompt |
| Add | `gen_fill:prompt_<prompt>` with crop region | Generate and composite a new object at a specified region |

### Additional AI effects

| Effect | Cloudinary effect | Description |
|---|---|---|
| Background remove | `background_removal` | Remove background, keep foreground |
| Background replace | `gen_background_replace:prompt_<prompt>` | Replace background with AI-generated scene |
| Generative fill | `gen_fill` / `gen_fill:prompt_<prompt>` | Extend or fill regions with generative AI |
| Blur (full frame) | `blur:<strength>` | Apply blur to the entire frame |
| Blur (region) | Mask overlay + `blur:1000` + `layer_apply` | Blur only the masked region (faces, plates) |
| Enhance | `enhance` | AI quality and detail improvement |
| Upscale | `upscale` | AI resolution upscaling |
| Restore | `gen_restore` | Fix compression artifacts and noise |
| Drop shadow | `background_removal` → `dropshadow:50` | Remove background then add drop shadow to subject |

### How it works

```
Frame JPEG → upload to Cloudinary → build transformation URL → download result → FFmpeg re-encode
```

Transformations are applied per frame across the selected frame range, then all processed frames are stitched back into a video and uploaded to Cloudinary for delivery.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/upload` | Create project, upload video to Cloudinary |
| POST | `/extract` | Extract frames with FFmpeg, run YOLO detection |
| GET | `/project/{id}/status` | Poll extraction status and frame count |
| GET | `/frame/{id}/{index}` | Serve a specific frame as JPEG |
| POST | `/detect` | On-demand YOLO detection for a single frame |
| POST | `/segment` | SAM 2 segmentation + propagation |
| POST | `/edit` | Apply batch edit rules across frame ranges |
| POST | `/render` | Re-encode video with FFmpeg, upload to Cloudinary |
