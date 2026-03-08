# FrameShift AI

By Bryan Lin, Justin Wang, Daniel Zhao, and Krish Punjabi.

AI-powered video editor. Upload a video, click an object, apply edits — changes propagate across every frame automatically.

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
| Storage & media | Cloudinary |
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
