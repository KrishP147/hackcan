# FrameShift Production Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `docs/frameshift-production-pipeline.md` (Document 1) end to end: morphing-free edit propagation on torch built-ins, all editing tools, temporally-coherent inpainting, dispatch rewrite, removals, extraction speedups, frontend updates, Modal deploy. Document 2 (CUDA) stays off this path — Backend B is a flag-guarded stub that falls back to Backend A.

**Architecture:** Edits fork into a deterministic path (per-frame OpenCV on stabilized SAM 2 masks) and a generative path (Gemini generates once at an anchor; RAFT star-warp composes pairwise flow into a single `A→t` displacement and takes ONE sample of the pristine anchor edit layer; forward-backward gating marks occlusions as holes; bidirectional blend + warp-then-refine re-anchoring bound drift). A shared flow-guided inpainter fills reveals/holes from neighboring frames instead of guessing per frame.

**Tech Stack:** PyTorch 2.10 (`grid_sample`, torchvision 0.25 RAFT), OpenCV, FastAPI, SAM 2, Gemini API (`gemini_service.edit_frame_with_reference` already exists), Next.js frontend, Modal deploy.

## Global Constraints

- **Interpreter:** `backend/venv` shebangs are broken. ALWAYS invoke via `./venv/bin/python -m <module>` from `backend/` (e.g. `./venv/bin/python -m pytest`), NEVER `./venv/bin/pytest` or `./venv/bin/uvicorn`.
- **Sampling convention (pin once, everywhere):** `grid_sample(..., mode="bilinear", padding_mode="zeros", align_corners=False)` with pixel-center `+0.5` normalization — exactly the `warp()` in spec §2.2. Doc 2 kernels must match it byte-for-byte later.
- **Flow direction naming:** `flow_fwd_%04d.npy` stores `F_{t→t+1}` (computed from pair `(t, t+1)`, indexed by `t`); `flow_bwd_%04d.npy` stores `F_{t+1→t}` (same index `t`). Backward-warping frame content from `t` to `t+1` samples with `F_{t+1→t}`.
- **Flags default off:** `USE_SYNTH = False`, `USE_CUDA_KERNEL = False` (`services/config.py`). Deployment never depends on them.
- **Device:** `cuda` if available, else `mps`, else `cpu`. All tests force `cpu` for determinism.
- **Working resolution:** frames extracted at `scale=-2:720`; H,W divisible by 8 required by RAFT (1280×720 OK; the `-2` keeps width even — pad to multiple of 8 inside flow_service if needed).
- **Storage layout (unchanged + new):** `projects/{id}/frames/frame_%04d.jpg`, `masks/mask_%04d.png` (0/255), NEW `flows/flow_{fwd,bwd}_%04d.npy`, status JSON via `project_manager`.
- **RIFE never touches object propagation.** It stays in the repo for slow-mo only.
- **Frame numbering:** ffmpeg emits `frame_0001.jpg` (1-based). Pair index `t` refers to `(frame_t, frame_{t+1})`.
- **Acceptance unit is the clip, never a single frame** (spec §11).

---

### Task 1: Test infrastructure + config flags

**Files:**
- Create: `backend/services/config.py`
- Create: `backend/tests/__init__.py`, `backend/tests/conftest.py`
- Modify: `backend/requirements.txt` (append `pytest==8.3.4`)

**Interfaces:**
- Produces: `config.USE_SYNTH: bool`, `config.USE_CUDA_KERNEL: bool`, `config.get_device() -> torch.device`; pytest runnable via `./venv/bin/python -m pytest tests/ -v`; conftest fixture `device` (cpu) and `tmp_project` (tmp dir with `frames/`, `masks/`, `flows/` subdirs).

- [ ] **Step 1: Write config + conftest**

```python
# backend/services/config.py
import os
import torch

USE_SYNTH = os.getenv("FRAMESHIFT_USE_SYNTH", "0") == "1"        # Backend B (Doc 2)
USE_CUDA_KERNEL = os.getenv("FRAMESHIFT_USE_CUDA_KERNEL", "0") == "1"

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

```python
# backend/tests/conftest.py
import pytest
import torch
from pathlib import Path

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    for sub in ("frames", "masks", "flows"):
        (tmp_path / sub).mkdir()
    return tmp_path
```

`backend/tests/__init__.py` is empty. Append to `backend/requirements.txt`:

```
pytest==8.3.4
```

- [ ] **Step 2: Install and verify collection**

Run: `cd backend && ./venv/bin/python -m pip install pytest==8.3.4 && ./venv/bin/python -m pytest tests/ -v`
Expected: `no tests ran` (collection succeeds, exit code 5 is fine).

- [ ] **Step 3: Commit**

```bash
git add backend/services/config.py backend/tests/ backend/requirements.txt
git commit -m "feat: test infra + backend-selection config flags"
```

---

### Task 2: Native-fps 720p extraction + drop per-frame Supabase upload

**Files:**
- Modify: `backend/services/ffmpeg_service.py` (`extract_frames` at :39, `encode_video` at :51)
- Modify: `backend/main.py` `_background_extract` (:73–120) — remove `storage_service.upload_frames` call, pass native fps, enforce clip cap
- Test: `backend/tests/test_ffmpeg_service.py`

**Interfaces:**
- Produces: `ffmpeg_service.probe_video(video_path: Path) -> dict` returning `{"fps": float, "duration": float, "width": int, "height": int}`; `extract_frames(video_path, output_dir, fps: float | None = None, max_height: int = 720) -> tuple[int, float]` returning `(frame_count, used_fps)` — `fps=None` means native rate; `encode_video(frames_dir, output_path, fps: float = 30)` accepts float fps.
- Consumes: nothing new. `main.py` stores `used_fps` in project status as `fps` so `/render` encodes at the true rate.

- [ ] **Step 1: Write failing tests** (generate a tiny 24fps synthetic clip with ffmpeg in the test)

```python
# backend/tests/test_ffmpeg_service.py
import subprocess
from pathlib import Path
import pytest
from services import ffmpeg_service

@pytest.fixture
def clip_24fps(tmp_path: Path) -> Path:
    out = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=1920x1080:rate=24",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)
    return out

def test_probe_reports_native_fps(clip_24fps):
    info = ffmpeg_service.probe_video(clip_24fps)
    assert abs(info["fps"] - 24.0) < 0.01
    assert info["width"] == 1920

def test_extract_native_fps_and_720p(clip_24fps, tmp_path):
    frames_dir = tmp_path / "frames"; frames_dir.mkdir()
    count, used_fps = ffmpeg_service.extract_frames(clip_24fps, frames_dir, fps=None)
    assert abs(used_fps - 24.0) < 0.01
    assert count == 24                      # 1s @ native 24fps, not forced 30
    import cv2
    img = cv2.imread(str(frames_dir / "frame_0001.jpg"))
    assert img.shape[0] == 720              # downscaled working frames
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ./venv/bin/python -m pytest tests/test_ffmpeg_service.py -v`
Expected: FAIL — `probe_video` doesn't exist / signature mismatch.

- [ ] **Step 3: Implement**

```python
# backend/services/ffmpeg_service.py — replace extract_frames, add probe_video
import json, subprocess
from pathlib import Path

def probe_video(video_path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", str(video_path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "fps": float(num) / float(den),
        "duration": float(s.get("duration") or 0),
        "width": int(s["width"]),
        "height": int(s["height"]),
    }

def extract_frames(video_path: Path, output_dir: Path,
                   fps: float | None = None, max_height: int = 720) -> tuple[int, float]:
    info = probe_video(video_path)
    used_fps = fps if fps else info["fps"]
    vf = []
    if fps:  # only resample when explicitly forced
        vf.append(f"fps={fps}")
    if info["height"] > max_height:
        vf.append(f"scale=-2:{max_height}")
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-qscale:v", "2", str(output_dir / "frame_%04d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(output_dir.glob("frame_*.jpg"))), used_fps
```

Update `encode_video` signature to `fps: float = 30` (pass-through to `-framerate`).

In `main.py` `_background_extract`: replace the `extract_frames(...)` call with the tuple form, store `fps=used_fps` via `project_manager.update_status`, delete the `storage_service.upload_frames(...)` call and its import if now unused, and reject clips where `probe_video(...)["duration"] > 15` with status `extract_status="error"`, `extract_error="Clip too long (max 15s)"`. In `/render` (`main.py:1383`), read `fps` from project status and pass it to `encode_video`.

- [ ] **Step 4: Run tests**

Run: `cd backend && ./venv/bin/python -m pytest tests/test_ffmpeg_service.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add backend/services/ffmpeg_service.py backend/main.py backend/tests/test_ffmpeg_service.py
git commit -m "feat: native-fps 720p extraction, 15s cap, drop per-frame storage uploads"
```

---

### Task 3: Flow primitives — warp, compose_flow, fb_check

**Files:**
- Create: `backend/services/flow_service.py`
- Test: `backend/tests/test_flow_service.py`

**Interfaces:**
- Produces (all torch, float32, no grad):
  - `warp(img: Tensor[N,C,H,W], flow: Tensor[N,2,H,W]) -> Tensor[N,C,H,W]` — backward bilinear sample, `out(x) = img(x + flow(x))`, zeros padding, the spec §2.2 convention.
  - `compose_flow(f_na: Tensor[N,2,H,W], f_step: Tensor[N,2,H,W]) -> Tensor[N,2,H,W]` — `F_{N→k-1}(x) = F_{N→k}(x) + sample(F_{k→k-1}, x + F_{N→k}(x))`, i.e. `f_na + warp(f_step, f_na)`.
  - `fb_check(f_fwd: Tensor[N,2,H,W], f_bwd: Tensor[N,2,H,W]) -> Tensor[N,1,H,W]` — validity in {0,1}: `‖f_fwd + warp(f_bwd, f_fwd)‖² ≤ 0.01·(‖f_fwd‖² + ‖f_bwd‖²) + 0.5`.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_flow_service.py
import torch
from services.flow_service import warp, compose_flow, fb_check

def const_flow(dx, dy, h=32, w=48):
    f = torch.zeros(1, 2, h, w)
    f[:, 0] = dx; f[:, 1] = dy
    return f

def test_warp_translation():
    img = torch.rand(1, 3, 32, 48)
    out = warp(img, const_flow(-5, 0))          # out(x) = img(x-5) → content shifts right
    assert torch.allclose(out[..., 5:], img[..., :-5], atol=1e-5)
    assert torch.all(out[..., :5].abs() < 1e-6)  # zeros padding at the seam

def test_warp_identity():
    img = torch.rand(1, 3, 32, 48)
    assert torch.allclose(warp(img, const_flow(0, 0)), img, atol=1e-5)

def test_compose_translations_add():
    f1, f2 = const_flow(3, 1), const_flow(2, -1)
    comp = compose_flow(f1, f2)
    assert torch.allclose(comp[:, 0, 8:-8, 8:-8], torch.tensor(5.0), atol=1e-4)
    assert torch.allclose(comp[:, 1, 8:-8, 8:-8], torch.tensor(0.0), atol=1e-4)

def test_fb_check_consistent_and_broken():
    v = fb_check(const_flow(5, 0), const_flow(-5, 0))
    assert v[:, :, 8:-8, 8:-8].min() == 1.0      # interior valid
    v_bad = fb_check(const_flow(5, 0), const_flow(5, 0))  # doesn't round-trip
    assert v_bad[:, :, 8:-8, 8:-8].max() == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ./venv/bin/python -m pytest tests/test_flow_service.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement** (verbatim spec §2.2/§2.3 math)

```python
# backend/services/flow_service.py
"""Backend A flow primitives. Sampling convention pinned here once:
grid_sample bilinear, padding zeros, align_corners=False, pixel centers +0.5.
Doc 2 CUDA kernels must match this byte-for-byte."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    n, _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device),
                            torch.arange(w, device=img.device), indexing="ij")
    base = torch.stack((xx, yy)).float()
    src = base[None] + flow
    gx = 2 * (src[:, 0] + 0.5) / w - 1
    gy = 2 * (src[:, 1] + 0.5) / h - 1
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)

@torch.no_grad()
def compose_flow(f_na: torch.Tensor, f_step: torch.Tensor) -> torch.Tensor:
    return f_na + warp(f_step, f_na)

@torch.no_grad()
def fb_check(f_fwd: torch.Tensor, f_bwd: torch.Tensor) -> torch.Tensor:
    err = (f_fwd + warp(f_bwd, f_fwd)).pow(2).sum(1, keepdim=True)
    bound = 0.01 * (f_fwd.pow(2).sum(1, keepdim=True)
                    + f_bwd.pow(2).sum(1, keepdim=True)) + 0.5
    return (err <= bound).float()
```

- [ ] **Step 4: Run tests** — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/flow_service.py backend/tests/test_flow_service.py
git commit -m "feat: flow primitives (warp/compose/fb-gate) with pinned sampling convention"
```

---

### Task 4: RAFT flow computation + disk cache

**Files:**
- Modify: `backend/services/flow_service.py` (append)
- Test: `backend/tests/test_flow_service.py` (append)

**Interfaces:**
- Produces:
  - `compute_flows(frames_dir: Path, flows_dir: Path, device: torch.device | None = None, num_flow_updates: int = 12) -> int` — computes fwd+bwd RAFT flow for every consecutive pair on **original footage**, saves `flows_dir/flow_fwd_%04d.npy` and `flow_bwd_%04d.npy` (float16 on disk, shape `(2,H,W)`), skips pairs already cached, returns pair count. Pads H,W to multiples of 8 internally and crops back.
  - `load_flow(flows_dir: Path, pair_index: int, direction: str) -> torch.Tensor[1,2,H,W]` (float32, cpu).
- Consumes: Task 3 primitives.

- [ ] **Step 1: Write failing test** (unit-level with a stubbed model; a real-RAFT smoke test marked slow)

```python
# append to backend/tests/test_flow_service.py
import numpy as np, cv2, pytest
from pathlib import Path
from services import flow_service

def _write_frames(frames_dir: Path, n=3, h=64, w=96):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), np.roll(base, i * 2, axis=1))

def test_compute_flows_caches_all_pairs(tmp_project, monkeypatch):
    _write_frames(tmp_project / "frames")
    class FakeRaft:  # returns constant 2px-right flow, matching the rolled frames
        def to(self, *a, **k): return self
        def eval(self): return self
        def __call__(self, a, b, num_flow_updates=12):
            n, _, h, w = a.shape
            f = torch.zeros(n, 2, h, w); f[:, 0] = -2.0
            return [f]
    import torch
    monkeypatch.setattr(flow_service, "_build_raft", lambda device: FakeRaft())
    pairs = flow_service.compute_flows(tmp_project / "frames", tmp_project / "flows",
                                       device=torch.device("cpu"))
    assert pairs == 2
    assert (tmp_project / "flows" / "flow_fwd_0001.npy").exists()
    assert (tmp_project / "flows" / "flow_bwd_0002.npy").exists()
    f = flow_service.load_flow(tmp_project / "flows", 1, "fwd")
    assert f.shape[1] == 2 and abs(f[:, 0].mean().item() + 2.0) < 1e-3

@pytest.mark.slow
def test_real_raft_smoke(tmp_project):
    _write_frames(tmp_project / "frames")
    import torch
    pairs = flow_service.compute_flows(tmp_project / "frames", tmp_project / "flows",
                                       device=torch.device("cpu"))
    assert pairs == 2  # downloads raft_large weights on first run
```

Register the marker in `backend/pytest.ini`:

```ini
[pytest]
markers =
    slow: needs model weights / real inference
addopts = -m "not slow"
```

- [ ] **Step 2: Run to verify failure** — `./venv/bin/python -m pytest tests/test_flow_service.py -v` → FAIL (`compute_flows` missing).

- [ ] **Step 3: Implement**

```python
# append to backend/services/flow_service.py
import numpy as np
from pathlib import Path

def _build_raft(device):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    return raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

def _load_frame_tensor(path: Path, device) -> torch.Tensor:
    import cv2
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 127.5 - 1.0  # [-1,1]
    return t.to(device)

def _pad8(t: torch.Tensor):
    h, w = t.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    return F.pad(t, (0, pw, 0, ph)), h, w

@torch.no_grad()
def compute_flows(frames_dir: Path, flows_dir: Path,
                  device: torch.device | None = None,
                  num_flow_updates: int = 12) -> int:
    from services.config import get_device
    device = device or get_device()
    flows_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    model = None
    pairs = 0
    for i in range(len(frames) - 1):
        idx = i + 1  # 1-based pair index = left frame number
        fwd_p = flows_dir / f"flow_fwd_{idx:04d}.npy"
        bwd_p = flows_dir / f"flow_bwd_{idx:04d}.npy"
        pairs += 1
        if fwd_p.exists() and bwd_p.exists():
            continue
        if model is None:
            model = _build_raft(device)
        a, h, w = _pad8(_load_frame_tensor(frames[i], device))
        b, _, _ = _pad8(_load_frame_tensor(frames[i + 1], device))
        fwd = model(a, b, num_flow_updates=num_flow_updates)[-1][..., :h, :w]
        bwd = model(b, a, num_flow_updates=num_flow_updates)[-1][..., :h, :w]
        np.save(fwd_p, fwd[0].cpu().numpy().astype(np.float16))
        np.save(bwd_p, bwd[0].cpu().numpy().astype(np.float16))
    return pairs

def load_flow(flows_dir: Path, pair_index: int, direction: str) -> torch.Tensor:
    arr = np.load(flows_dir / f"flow_{direction}_{pair_index:04d}.npy").astype(np.float32)
    return torch.from_numpy(arr)[None]
```

- [ ] **Step 4: Run tests** — Expected: cache test PASS, slow test deselected.

- [ ] **Step 5: Run the slow test once locally** — `./venv/bin/python -m pytest tests/test_flow_service.py -m slow -v` → PASS (validates real RAFT + weights download).

- [ ] **Step 6: Commit**

```bash
git add backend/services/flow_service.py backend/tests/test_flow_service.py backend/pytest.ini
git commit -m "feat: RAFT pairwise flow with per-project disk cache"
```

---

### Task 5: Mask conditioning (§2.7)

**Files:**
- Create: `backend/services/mask_service.py`
- Test: `backend/tests/test_mask_service.py`

**Interfaces:**
- Produces: `stabilize_masks(masks_dir: Path, out_dir: Path | None = None) -> int` — for every `mask_%04d.png`: morphological close (3×3), then 2-frame EMA on mask alpha (`m_t = 0.6·m_t + 0.4·m_{t-1}`), rebinarize at 127, overwrite in place (or `out_dir`), returns count. `load_mask_alpha(masks_dir: Path, frame_index: int, feather_px: int = 0) -> np.ndarray[H,W] float32 in [0,1]` — optional Gaussian feather of the boundary (used by compositing §2.5 and per-tool alphas).

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_mask_service.py
import numpy as np, cv2
from services import mask_service

def _disk(h, w, cx, cy, r):
    yy, xx = np.mgrid[:h, :w]
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8) * 255

def test_stabilize_closes_pinholes_and_smooths_jitter(tmp_project):
    masks = tmp_project / "masks"
    m1 = _disk(64, 64, 32, 32, 20); m1[32, 32] = 0            # pinhole
    m2 = _disk(64, 64, 33, 32, 20)                             # 1px jitter
    cv2.imwrite(str(masks / "mask_0001.png"), m1)
    cv2.imwrite(str(masks / "mask_0002.png"), m2)
    n = mask_service.stabilize_masks(masks)
    assert n == 2
    s1 = cv2.imread(str(masks / "mask_0001.png"), 0)
    assert s1[32, 32] == 255                                   # pinhole closed
    s2 = cv2.imread(str(masks / "mask_0002.png"), 0)
    assert set(np.unique(s2)) <= {0, 255}                      # still binary

def test_feathered_alpha_is_soft_at_boundary(tmp_project):
    masks = tmp_project / "masks"
    cv2.imwrite(str(masks / "mask_0001.png"), _disk(64, 64, 32, 32, 20))
    a = mask_service.load_mask_alpha(masks, 1, feather_px=5)
    assert a.max() == 1.0 and a.min() == 0.0
    assert ((a > 0.05) & (a < 0.95)).sum() > 50                # soft transition band exists
```

- [ ] **Step 2: Run to verify failure** — FAIL, module missing.

- [ ] **Step 3: Implement**

```python
# backend/services/mask_service.py
import cv2
import numpy as np
from pathlib import Path

_KERNEL = np.ones((3, 3), np.uint8)

def stabilize_masks(masks_dir: Path, out_dir: Path | None = None) -> int:
    out_dir = out_dir or masks_dir
    paths = sorted(masks_dir.glob("mask_*.png"))
    prev = None
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _KERNEL)
        if prev is not None:
            m = 0.6 * m + 0.4 * prev
        prev = m
        binary = (m >= 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / p.name), binary)
    return len(paths)

def load_mask_alpha(masks_dir: Path, frame_index: int, feather_px: int = 0) -> np.ndarray:
    p = masks_dir / f"mask_{frame_index:04d}.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if feather_px > 0:
        k = feather_px * 2 + 1
        blurred = cv2.GaussianBlur(a, (k, k), 0)
        a = np.where((a > 0) & (a < 1), a, blurred)  # keep hard core, soften only boundary
        a = np.maximum(np.minimum(a, 1.0), 0.0)
        core = cv2.erode((a >= 0.999).astype(np.uint8), _KERNEL, iterations=feather_px)
        a = np.maximum(a * 0 + blurred, core.astype(np.float32))
        a = np.clip(a, 0.0, 1.0)
    return a
```

Note: the feather implementation must satisfy the test — full 1.0 deep inside, 0.0 far outside, soft in between. If the double-max approach reads muddy, simplify to `a = np.clip(blurred, 0, 1); a[core == 1] = 1.0` — the test is the contract.

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Wire into segmentation:** in `main.py` `_background_segment_and_propagate` (:190) call `mask_service.stabilize_masks(masks_dir)` after masks are saved (single-frame path: it's a no-op close). In `sam2_service.propagate_masks` callers, stabilize after propagation completes.

- [ ] **Step 6: Commit**

```bash
git add backend/services/mask_service.py backend/tests/test_mask_service.py backend/main.py
git commit -m "feat: mask stabilization (close + EMA) and feathered alpha loader"
```

---

### Task 6: Star-warp propagation engine — EditLayer, forward star warp, composite (§2.2, §2.3, §2.5)

**Files:**
- Create: `backend/services/propagation_service.py`
- Test: `backend/tests/test_propagation_service.py`

**Interfaces:**
- Produces:
  - `@dataclass EditLayer: rgb: np.ndarray[H,W,3] uint8; alpha: np.ndarray[H,W] float32; validity: np.ndarray[H,W] float32`
  - `make_anchor_layer(edited_frame: np.ndarray[H,W,3], mask_alpha: np.ndarray[H,W]) -> EditLayer` (validity = alpha > 0).
  - `star_warp(anchor: EditLayer, anchor_index: int, target_indices: list[int], flows_dir: Path, device) -> dict[int, EditLayer]` — composes pairwise flow `A→t` per spec §2.2 (ONE sample of pristine anchor RGBA+validity per target), applies FB gate on the composed round-trip (§2.3): gated pixels get validity 0. Works both directions (targets before or after anchor).
  - `composite(frame: np.ndarray[H,W,3] uint8, layer: EditLayer, mask_alpha: np.ndarray[H,W]) -> np.ndarray[H,W,3] uint8` — `out = α·E + (1−α)·I` where `α = mask_alpha · layer.alpha · (layer.validity ≥ 0.5)`.
- Consumes: `flow_service.warp/compose_flow/fb_check/load_flow` (Task 3/4 signatures).

**Direction bookkeeping (the subtle part, spelled out):** to place anchor content at target `t > A`, you need, for each target pixel, where it came from in the anchor — i.e. the *backward* displacement `F_{t→A}`. Build it by composing per-pair **bwd** flows: start `F = flow_bwd[t-1]` (maps `t → t-1`), then for `k = t-1 … A+1`: `F = compose_flow(F, flow_bwd[k-1])`. For `t < A`, compose per-pair **fwd** flows symmetrically. The FB gate for target `t` checks the composed `F_{t→A}` against the composed `F_{A→t}` (built the same way in the opposite direction).

- [ ] **Step 1: Write failing test** (synthetic translating scene, constant flows on disk)

```python
# backend/tests/test_propagation_service.py
import numpy as np, torch
from services.propagation_service import EditLayer, make_anchor_layer, star_warp, composite

H, W, SHIFT = 48, 64, 2  # scene translates right 2px/frame

def _write_const_flows(flows_dir, n_pairs, dx):
    for i in range(1, n_pairs + 1):
        fwd = np.zeros((2, H, W), np.float16); fwd[0] = dx    # F_{t→t+1}
        bwd = np.zeros((2, H, W), np.float16); bwd[0] = -dx   # F_{t+1→t}
        np.save(flows_dir / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(flows_dir / f"flow_bwd_{i:04d}.npy", bwd)

def test_star_warp_carries_one_appearance(tmp_project):
    _write_const_flows(tmp_project / "flows", 4, SHIFT)
    rgb = np.zeros((H, W, 3), np.uint8); rgb[20:30, 10:20] = (255, 0, 0)  # red box at anchor
    alpha = np.zeros((H, W), np.float32); alpha[20:30, 10:20] = 1.0
    anchor = make_anchor_layer(rgb, alpha)
    layers = star_warp(anchor, anchor_index=1, target_indices=[2, 3, 5],
                       flows_dir=tmp_project / "flows", device=torch.device("cpu"))
    l5 = layers[5]                          # 4 steps of +2px → box at x=[18,28)
    assert l5.rgb[24, 22, 0] > 200          # red carried, not re-invented
    assert l5.alpha[24, 22] > 0.9
    assert l5.alpha[24, 12] < 0.1           # old location vacated
    assert l5.validity[24, 22] > 0.9

def test_composite_blends_only_valid_masked_pixels(tmp_project):
    frame = np.full((H, W, 3), 100, np.uint8)
    rgb = np.zeros((H, W, 3), np.uint8); rgb[:, :, 2] = 255
    alpha = np.ones((H, W), np.float32)
    layer = EditLayer(rgb=rgb, alpha=alpha, validity=np.ones((H, W), np.float32))
    layer.validity[:, W // 2:] = 0.0        # right half gated out
    mask = np.ones((H, W), np.float32)
    out = composite(frame, layer, mask)
    assert out[10, 5, 2] == 255             # valid → edit shows
    assert out[10, W - 5, 2] == 100         # gated → original footage
```

- [ ] **Step 2: Run to verify failure** — FAIL, module missing.

- [ ] **Step 3: Implement**

```python
# backend/services/propagation_service.py
"""Backend A star-warp propagation (Doc 1 §2). Carry ONE appearance:
compose pairwise flow into a single A→t displacement, sample pristine anchor once."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from services import flow_service

@dataclass
class EditLayer:
    rgb: np.ndarray        # (H,W,3) uint8
    alpha: np.ndarray      # (H,W) float32 [0,1]
    validity: np.ndarray   # (H,W) float32 [0,1]

def make_anchor_layer(edited_frame: np.ndarray, mask_alpha: np.ndarray) -> EditLayer:
    return EditLayer(rgb=edited_frame.copy(),
                     alpha=mask_alpha.astype(np.float32).copy(),
                     validity=(mask_alpha > 0).astype(np.float32))

def _compose_chain(flows_dir: Path, start: int, end: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Composed displacement F_{end→start} (for sampling) and F_{start→end} (for FB gate).
    start=anchor A, end=target t. Handles t>A and t<A."""
    if end > start:
        back = flow_service.load_flow(flows_dir, end - 1, "bwd").to(device)   # t→t-1
        for k in range(end - 1, start, -1):
            back = flow_service.compose_flow(back, flow_service.load_flow(flows_dir, k - 1, "bwd").to(device))
        fwd = flow_service.load_flow(flows_dir, start, "fwd").to(device)      # A→A+1
        for k in range(start + 1, end):
            fwd = flow_service.compose_flow(fwd, flow_service.load_flow(flows_dir, k, "fwd").to(device))
        return back, fwd
    else:  # end < start: sample direction uses fwd pairs, gate uses bwd pairs
        back = flow_service.load_flow(flows_dir, end, "fwd").to(device)       # t→t+1
        for k in range(end + 1, start):
            back = flow_service.compose_flow(back, flow_service.load_flow(flows_dir, k, "fwd").to(device))
        fwd = flow_service.load_flow(flows_dir, start - 1, "bwd").to(device)  # A→A-1
        for k in range(start - 1, end, -1):
            fwd = flow_service.compose_flow(fwd, flow_service.load_flow(flows_dir, k - 1, "bwd").to(device))
        return back, fwd

@torch.no_grad()
def star_warp(anchor: EditLayer, anchor_index: int, target_indices: list[int],
              flows_dir: Path, device: torch.device) -> dict[int, EditLayer]:
    payload = np.concatenate([anchor.rgb.astype(np.float32),
                              anchor.alpha[..., None] * 255.0,
                              anchor.validity[..., None] * 255.0], axis=2)  # (H,W,5)
    src = torch.from_numpy(payload).permute(2, 0, 1)[None].to(device)
    out: dict[int, EditLayer] = {}
    for t in sorted(target_indices):
        if t == anchor_index:
            out[t] = EditLayer(anchor.rgb.copy(), anchor.alpha.copy(), anchor.validity.copy())
            continue
        f_ta, f_at = _compose_chain(flows_dir, anchor_index, t, device)
        warped = flow_service.warp(src, f_ta)[0].permute(1, 2, 0).cpu().numpy()
        valid = flow_service.fb_check(f_ta, f_at)[0, 0].cpu().numpy()      # gate on composed flows
        out[t] = EditLayer(
            rgb=np.clip(warped[..., :3], 0, 255).astype(np.uint8),
            alpha=np.clip(warped[..., 3] / 255.0, 0, 1),
            validity=np.clip(warped[..., 4] / 255.0, 0, 1) * valid,
        )
    return out

def composite(frame: np.ndarray, layer: EditLayer, mask_alpha: np.ndarray) -> np.ndarray:
    a = (mask_alpha * layer.alpha * (layer.validity >= 0.5))[..., None].astype(np.float32)
    out = a * layer.rgb.astype(np.float32) + (1 - a) * frame.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests** — Expected: PASS. (If the box lands 1px off, re-check flow sign convention against the Task 3 translation test before touching thresholds — direction bugs, not tolerance, are the likely cause.)

- [ ] **Step 5: Commit**

```bash
git add backend/services/propagation_service.py backend/tests/test_propagation_service.py
git commit -m "feat: star-warp propagation engine with composed-flow FB gating"
```

---

### Task 7: Bidirectional blend + re-anchor triggers (§2.4, §2.6)

**Files:**
- Modify: `backend/services/propagation_service.py` (append)
- Test: `backend/tests/test_propagation_service.py` (append)

**Interfaces:**
- Produces:
  - `blend_bidirectional(past: EditLayer, fut: EditLayer, dist_past: int, dist_fut: int, eps: float = 1e-3) -> EditLayer` — per-pixel weights `w = validity / (dist + eps)` (spec §2.4); pixels where both validities < 0.5 keep validity 0 (caller sends them to the inpainter).
  - `needs_reanchor(layer: EditLayer, mask_alpha: np.ndarray, anchor_mask_area: float, frames_since_anchor: int) -> bool` — any trigger (§2.6): mean validity inside mask < 0.7; `|mask_area/anchor_area − 1| > 0.3`; `frames_since_anchor ≥ 60`.
- Consumes: `EditLayer` from Task 6.

- [ ] **Step 1: Write failing tests**

```python
# append to backend/tests/test_propagation_service.py
from services.propagation_service import blend_bidirectional, needs_reanchor

def _layer(val_left, val_right, red=255):
    rgb = np.zeros((H, W, 3), np.uint8); rgb[:, :, 0] = red
    alpha = np.ones((H, W), np.float32)
    v = np.zeros((H, W), np.float32); v[:, :W//2] = val_left; v[:, W//2:] = val_right
    return EditLayer(rgb, alpha, v)

def test_bidirectional_fills_disocclusion_from_other_side():
    past = _layer(1.0, 0.0, red=200)   # right half occluded looking forward
    fut = _layer(1.0, 1.0, red=100)    # visible looking back
    out = blend_bidirectional(past, fut, dist_past=2, dist_fut=2)
    assert out.rgb[10, W - 5, 0] == 100          # right half sourced from future
    assert 100 < out.rgb[10, 5, 0] < 200         # left half blends both
    assert out.validity[10, W - 5] > 0.5

def test_reanchor_triggers():
    good = _layer(1.0, 1.0)
    mask = np.ones((H, W), np.float32)
    assert not needs_reanchor(good, mask, anchor_mask_area=mask.sum(), frames_since_anchor=10)
    assert needs_reanchor(_layer(0.4, 0.4), mask, mask.sum(), 10)          # validity collapse
    assert needs_reanchor(good, mask, anchor_mask_area=mask.sum() * 2, frames_since_anchor=10)  # area jump
    assert needs_reanchor(good, mask, mask.sum(), frames_since_anchor=60)  # chain cap
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# append to backend/services/propagation_service.py
def blend_bidirectional(past: EditLayer, fut: EditLayer,
                        dist_past: int, dist_fut: int, eps: float = 1e-3) -> EditLayer:
    wp = past.validity / (dist_past + eps)
    wf = fut.validity / (dist_fut + eps)
    tot = wp + wf
    safe = np.where(tot > 0, tot, 1.0)[..., None]
    rgb = (wp[..., None] * past.rgb + wf[..., None] * fut.rgb) / safe
    alpha = (wp * past.alpha + wf * fut.alpha) / safe[..., 0]
    validity = np.maximum(past.validity, fut.validity)
    validity = np.where(np.maximum(past.validity, fut.validity) >= 0.5, validity, 0.0)
    return EditLayer(np.clip(rgb, 0, 255).astype(np.uint8),
                     np.clip(alpha, 0, 1), validity.astype(np.float32))

def needs_reanchor(layer: EditLayer, mask_alpha: np.ndarray,
                   anchor_mask_area: float, frames_since_anchor: int) -> bool:
    inside = mask_alpha > 0.5
    if inside.any() and layer.validity[inside].mean() < 0.7:
        return True
    area = float(inside.sum())
    if anchor_mask_area > 0 and abs(area / anchor_mask_area - 1.0) > 0.3:
        return True
    return frames_since_anchor >= 60
```

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/propagation_service.py backend/tests/test_propagation_service.py
git commit -m "feat: bidirectional blend and warp-then-refine re-anchor triggers"
```

---

### Task 8: Temporally-coherent inpainter (§4)

**Files:**
- Create: `backend/services/inpaint_service.py`
- Test: `backend/tests/test_inpaint_service.py`

**Interfaces:**
- Produces: `inpaint_video(frames: dict[int, np.ndarray], hole_masks: dict[int, np.ndarray], flows_dir: Path, device: torch.device, max_donor_dist: int = 8) -> dict[int, np.ndarray]` — keys are frame indices; `hole_masks` uint8/bool (True = hole). Ladder per spec §4: (1) flow-guided fill — for each hole pixel in frame `t`, warp in candidate pixels from donor frames `t±k` (nearest first) using composed flow, take the first donor whose FB-gated validity passes and whose donor pixel is NOT itself in the donor's hole; blend multiple donors by `1/(dist+ε)`; (2) TELEA residual on never-seen pixels, then 2-frame EMA over consecutive TELEA patches so the residual doesn't shimmer.
- Consumes: `flow_service` primitives, `_compose_chain` (import from `propagation_service`).

- [ ] **Step 1: Write failing test** (static textured background + moving hole → flow-guided fill must recover the background exactly; a hole present in ALL frames → TELEA residual fills it)

```python
# backend/tests/test_inpaint_service.py
import numpy as np, torch
from services.inpaint_service import inpaint_video

H, W = 48, 64

def _zero_flows(flows_dir, n_pairs):
    z = np.zeros((2, H, W), np.float16)
    for i in range(1, n_pairs + 1):
        np.save(flows_dir / f"flow_fwd_{i:04d}.npy", z)
        np.save(flows_dir / f"flow_bwd_{i:04d}.npy", z)

def test_flow_guided_fill_recovers_static_background(tmp_project):
    _zero_flows(tmp_project / "flows", 2)
    rng = np.random.default_rng(1)
    bg = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    frames, holes = {}, {}
    for t, x0 in [(1, 10), (2, 25), (3, 40)]:      # hole slides across static bg
        holes[t] = np.zeros((H, W), bool); holes[t][20:30, x0:x0 + 10] = True
        f = bg.copy(); f[holes[t]] = 0
        frames[t] = f
    out = inpaint_video(frames, holes, tmp_project / "flows", torch.device("cpu"))
    # frame 2's hole is visible in frames 1 and 3 → recovered near-exactly
    assert np.abs(out[2][20:30, 25:35].astype(int) - bg[20:30, 25:35].astype(int)).mean() < 2

def test_never_seen_pixels_fall_back_to_telea(tmp_project):
    _zero_flows(tmp_project / "flows", 1)
    bg = np.full((H, W, 3), 128, np.uint8)
    hole = np.zeros((H, W), bool); hole[20:30, 20:30] = True   # same hole every frame
    frames = {1: bg.copy(), 2: bg.copy()}
    for f in frames.values(): f[hole] = 0
    out = inpaint_video(frames, {1: hole, 2: hole}, tmp_project / "flows", torch.device("cpu"))
    assert np.abs(out[1][22:28, 22:28].astype(int) - 128).mean() < 10   # TELEA on flat bg ≈ flat
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# backend/services/inpaint_service.py
"""Temporally-coherent inpainting (Doc 1 §4): source vacated background from
neighboring frames via cached flow; TELEA only for never-seen residual, EMA'd."""
from pathlib import Path
import cv2
import numpy as np
import torch
from services import flow_service
from services.propagation_service import _compose_chain

@torch.no_grad()
def inpaint_video(frames: dict[int, np.ndarray], hole_masks: dict[int, np.ndarray],
                  flows_dir: Path, device: torch.device,
                  max_donor_dist: int = 8) -> dict[int, np.ndarray]:
    indices = sorted(frames)
    lo, hi = indices[0], indices[-1]
    out: dict[int, np.ndarray] = {}
    prev_telea: np.ndarray | None = None
    prev_telea_mask: np.ndarray | None = None
    for t in indices:
        frame = frames[t].astype(np.float32)
        hole = hole_masks[t].astype(bool)
        acc = np.zeros_like(frame)
        wsum = np.zeros(frame.shape[:2], np.float32)
        for k in range(1, max_donor_dist + 1):
            for d in (t - k, t + k):
                if d < lo or d > hi or not hole.any():
                    continue
                f_td, f_dt = _compose_chain(flows_dir, d, t, device)  # F_{t→d} sample dir
                donor = torch.from_numpy(frames[d].astype(np.float32)).permute(2, 0, 1)[None].to(device)
                donor_hole = torch.from_numpy((~hole_masks[d]).astype(np.float32))[None, None].to(device)
                warped = flow_service.warp(donor, f_td)[0].permute(1, 2, 0).cpu().numpy()
                usable = flow_service.warp(donor_hole, f_td)[0, 0].cpu().numpy()
                valid = flow_service.fb_check(f_td, f_dt)[0, 0].cpu().numpy()
                w = (usable > 0.99).astype(np.float32) * valid / (k + 1e-3)
                w = np.where(hole, w, 0.0)
                acc += w[..., None] * warped
                wsum += w
        filled = frame.copy()
        got = wsum > 0
        filled[got] = acc[got] / wsum[got][..., None]
        residual = hole & ~got
        if residual.any():
            telea = cv2.inpaint(np.clip(filled, 0, 255).astype(np.uint8),
                                residual.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32)
            if prev_telea is not None and prev_telea_mask is not None:
                shared = residual & prev_telea_mask
                telea[shared] = 0.5 * telea[shared] + 0.5 * prev_telea[shared]  # EMA: no shimmer
            filled[residual] = telea[residual]
            prev_telea, prev_telea_mask = telea, residual
        else:
            prev_telea, prev_telea_mask = None, None
        out[t] = np.clip(filled, 0, 255).astype(np.uint8)
    return out
```

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/inpaint_service.py backend/tests/test_inpaint_service.py
git commit -m "feat: flow-guided temporal inpainter with EMA'd TELEA residual"
```

---

### Task 9: Deterministic in-mask tools — color_pop, glow (+ keep recolor/blur) (§5.1)

**Files:**
- Modify: `backend/services/local_edit_service.py` (append two functions; delete `apply_enhance`, `apply_upscale`, `apply_restore` at :153–:186)
- Test: `backend/tests/test_local_edit_service.py`

**Interfaces:**
- Produces: `apply_color_pop(frame_path: Path, mask_path: Path) -> None` — desaturate OUTSIDE the mask (α = 1−M, feathered); `apply_glow(frame_path: Path, mask_path: Path, intensity: float = 0.6, radius: int = 21) -> None` — screen-blend a Gaussian-blurred bright copy of the masked object over the frame (halo extends outside the mask). Both overwrite the frame in place like the existing `apply_recolor`.
- Consumes: `mask_service.load_mask_alpha(masks_dir, idx, feather_px=5)` — but to match existing `local_edit_service` style (paths, not dirs) these take `mask_path` and feather internally.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_local_edit_service.py
import cv2, numpy as np
from pathlib import Path
from services import local_edit_service

def _setup(tmp_path):
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[:, :, 2] = 200                              # red-ish everywhere (BGR)
    fp = tmp_path / "frame_0001.jpg"; cv2.imwrite(str(fp), frame)
    mask = np.zeros((64, 64), np.uint8); mask[20:44, 20:44] = 255
    mp = tmp_path / "mask_0001.png"; cv2.imwrite(str(mp), mask)
    return fp, mp

def test_color_pop_desaturates_outside_only(tmp_path):
    fp, mp = _setup(tmp_path)
    local_edit_service.apply_color_pop(fp, mp)
    out = cv2.imread(str(fp))
    b, g, r = out[32, 32].astype(int)                 # inside: still colorful
    assert r - b > 100
    b, g, r = out[5, 5].astype(int)                   # outside: gray (channels equal-ish)
    assert abs(r - b) < 20 and abs(r - g) < 20

def test_glow_brightens_ring_outside_mask(tmp_path):
    fp, mp = _setup(tmp_path)
    before = cv2.imread(str(fp)).astype(int)
    local_edit_service.apply_glow(fp, mp)
    after = cv2.imread(str(fp)).astype(int)
    assert after[18, 32].sum() > before[18, 32].sum() + 30   # halo just outside mask
    assert abs(after[2, 2].sum() - before[2, 2].sum()) < 30  # far field untouched
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# append to backend/services/local_edit_service.py
def _feathered_alpha(mask_path: Path, shape: tuple, feather_px: int = 5) -> np.ndarray:
    mask = _load_mask(mask_path, shape)
    a = mask.astype(np.float32)
    a = a / a.max() if a.max() > 0 else a
    k = feather_px * 2 + 1
    return cv2.GaussianBlur(a, (k, k), 0)

def apply_color_pop(frame_path: Path, mask_path: Path) -> None:
    img = cv2.imread(str(frame_path))
    a = _feathered_alpha(mask_path, img.shape)[..., None]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = a * img.astype(np.float32) + (1 - a) * gray3
    cv2.imwrite(str(frame_path), np.clip(out, 0, 255).astype(np.uint8))

def apply_glow(frame_path: Path, mask_path: Path,
               intensity: float = 0.6, radius: int = 21) -> None:
    img = cv2.imread(str(frame_path)).astype(np.float32)
    a = _feathered_alpha(mask_path, img.shape)
    obj = img * a[..., None]
    halo = cv2.GaussianBlur(obj, (radius, radius), 0) * intensity
    out = 255 - (255 - img) * (255 - halo) / 255          # screen blend
    cv2.imwrite(str(frame_path), np.clip(out, 0, 255).astype(np.uint8))
```

Delete `apply_enhance`, `apply_upscale`, `apply_restore` from this file (check `main.py` callers — they go away in Task 13).

- [ ] **Step 4: Run tests** — Expected: PASS. Run full suite too: `./venv/bin/python -m pytest tests/ -v`.

- [ ] **Step 5: Commit**

```bash
git add backend/services/local_edit_service.py backend/tests/test_local_edit_service.py
git commit -m "feat: color_pop and glow deterministic tools; drop enhance/upscale/restore"
```

---

### Task 10: delete / resize-reveal / move on the inpainter (§5.2–§5.4)

**Files:**
- Create: `backend/services/object_tools.py`
- Test: `backend/tests/test_object_tools.py`

**Interfaces:**
- Produces (all operate on whole frame ranges so they can share one `inpaint_video` call):
  - `apply_delete_range(project_dir: Path, frame_indices: list[int], device) -> None` — holes = stabilized masks; fill via `inpaint_service.inpaint_video`; overwrite frames.
  - `apply_resize_range(project_dir: Path, frame_indices: list[int], scale: float, device) -> None` — per frame: cutout scaled about mask centroid (reuse the affine logic from `local_edit_service.apply_resize` :53); if `scale < 1`, reveal ring = `M_t ∧ ¬M_t_scaled` collected across frames → one `inpaint_video` call → paste scaled cutout over inpainted plate with feathered alpha.
  - `apply_move_range(project_dir: Path, frame_indices: list[int], offsets: dict[int, tuple[int,int]], device) -> None` — per frame `t`: cutout via `M_t` from ORIGINAL frame; vacated hole = `M_t ∧ ¬shifted(M_t)`; inpaint all vacated holes in one call; paste cutout at `(dx,dy)_t` with feathered alpha. `offsets` carries per-frame linear interpolation done by the caller (dispatch).
- Consumes: `inpaint_service.inpaint_video` (Task 8), `mask_service.load_mask_alpha` (Task 5).

- [ ] **Step 1: Write failing test** (move: static background, object translated; vacated region must match background; object must appear at target)

```python
# backend/tests/test_object_tools.py
import cv2, numpy as np, torch
from services import object_tools

H, W = 48, 64

def _project(tmp_project, n=3):
    z = np.zeros((2, H, W), np.float16)
    for i in range(1, n):
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", z)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", z)
    bg = np.full((H, W, 3), 90, np.uint8)
    for t in range(1, n + 1):
        f = bg.copy(); f[20:30, 10:20] = (0, 0, 255)          # red box (BGR)
        cv2.imwrite(str(tmp_project / "frames" / f"frame_{t:04d}.jpg"), f)
        m = np.zeros((H, W), np.uint8); m[20:30, 10:20] = 255
        cv2.imwrite(str(tmp_project / "masks" / f"mask_{t:04d}.png"), m)
    return tmp_project

def test_move_transports_object_and_fills_vacated(tmp_project):
    p = _project(tmp_project)
    object_tools.apply_move_range(p, [1, 2, 3], {1: (20, 0), 2: (20, 0), 3: (20, 0)},
                                  torch.device("cpu"))
    out = cv2.imread(str(p / "frames" / "frame_0002.jpg"))
    assert out[24, 35, 2] > 180 and out[24, 35, 0] < 80       # object at new spot
    assert abs(int(out[24, 12, 2]) - 90) < 25                  # vacated ≈ background
    assert abs(int(out[24, 12, 0]) - 90) < 25

def test_delete_removes_object(tmp_project):
    p = _project(tmp_project)
    object_tools.apply_delete_range(p, [1, 2, 3], torch.device("cpu"))
    out = cv2.imread(str(p / "frames" / "frame_0002.jpg"))
    assert abs(int(out[24, 14, 2]) - 90) < 25                  # red box gone → bg
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# backend/services/object_tools.py
"""delete / resize / move — deterministic transports riding the §4 inpainter."""
from pathlib import Path
import cv2
import numpy as np
import torch
from services import inpaint_service, mask_service

def _load(project_dir: Path, indices):
    frames = {t: cv2.imread(str(project_dir / "frames" / f"frame_{t:04d}.jpg")) for t in indices}
    masks = {t: mask_service.load_mask_alpha(project_dir / "masks", t) for t in indices}
    return frames, masks

def _save(project_dir: Path, frames: dict):
    for t, img in frames.items():
        cv2.imwrite(str(project_dir / "frames" / f"frame_{t:04d}.jpg"), img)

def apply_delete_range(project_dir: Path, frame_indices, device) -> None:
    frames, masks = _load(project_dir, frame_indices)
    holes = {t: masks[t] > 0.5 for t in frame_indices}
    _save(project_dir, inpaint_service.inpaint_video(frames, holes,
                                                     project_dir / "flows", device))

def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    m = np.zeros_like(mask)
    h, w = mask.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    m[ys0:ys1, xs0:xs1] = mask[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return m

def apply_move_range(project_dir: Path, frame_indices, offsets: dict, device) -> None:
    frames, masks = _load(project_dir, frame_indices)
    cutouts = {t: frames[t].copy() for t in frame_indices}
    holes = {}
    for t in frame_indices:
        dx, dy = offsets[t]
        moved = _shift_mask(masks[t], dx, dy)
        holes[t] = (masks[t] > 0.5) & ~(moved > 0.5)   # vacated minus new footprint
    plates = inpaint_service.inpaint_video(frames, holes, project_dir / "flows", device)
    out = {}
    for t in frame_indices:
        dx, dy = offsets[t]
        alpha = mask_service.load_mask_alpha(project_dir / "masks", t, feather_px=3)
        moved_alpha = _shift_mask(alpha, dx, dy)[..., None]
        moved_cut = _shift_mask_rgb(cutouts[t], dx, dy)
        plate = plates[t].astype(np.float32)
        out[t] = np.clip(moved_alpha * moved_cut + (1 - moved_alpha) * plate,
                         0, 255).astype(np.uint8)
    _save(project_dir, out)

def _shift_mask_rgb(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.stack([_shift_mask(img[..., c].astype(np.float32), dx, dy)
                     for c in range(img.shape[2])], axis=2)

def apply_resize_range(project_dir: Path, frame_indices, scale: float, device) -> None:
    frames, masks = _load(project_dir, frame_indices)
    holes = {}
    scaled = {}
    for t in frame_indices:
        m = (masks[t] > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            holes[t] = np.zeros_like(m, bool); scaled[t] = None; continue
        cx, cy = xs.mean(), ys.mean()
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        h, w = m.shape
        s_alpha = cv2.warpAffine(mask_service.load_mask_alpha(project_dir / "masks", t, feather_px=3),
                                 M, (w, h))
        s_rgb = cv2.warpAffine(frames[t], M, (w, h))
        scaled[t] = (s_rgb, s_alpha)
        holes[t] = (m > 0) & ~(s_alpha > 0.5) if scale < 1.0 else np.zeros_like(m, bool)
    plates = inpaint_service.inpaint_video(frames, holes, project_dir / "flows", device) \
        if any(h.any() for h in holes.values()) else {t: frames[t] for t in frame_indices}
    out = {}
    for t in frame_indices:
        if scaled[t] is None:
            out[t] = frames[t]; continue
        s_rgb, s_alpha = scaled[t]
        a = s_alpha[..., None]
        out[t] = np.clip(a * s_rgb.astype(np.float32)
                         + (1 - a) * plates[t].astype(np.float32), 0, 255).astype(np.uint8)
    _save(project_dir, out)
```

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/object_tools.py backend/tests/test_object_tools.py
git commit -m "feat: delete/move/resize riding the temporal inpainter"
```

---

### Task 11: `replace` — full generative engine with re-anchoring (§5.6)

**Files:**
- Create: `backend/services/replace_tool.py`
- Test: `backend/tests/test_replace_tool.py`

**Interfaces:**
- Produces: `async apply_replace_range(project_dir: Path, frame_indices: list[int], anchor_index: int, prompt: str, device, generate=None) -> None`. `generate` is an injectable async callable `(frame_path, prompt, reference_frame_path, mask_path) -> bytes` defaulting to `gemini_service.edit_frame_with_reference` — injection keeps tests offline. Algorithm:
  1. Generate ONCE at anchor (`generate(anchor_frame, prompt, None, anchor_mask)`), decode bytes → `make_anchor_layer` with the anchor's feathered mask alpha.
  2. Walk targets outward from the anchor in segments. For the current anchor, `star_warp` toward the segment's frames in order; after each frame, evaluate `needs_reanchor(layer, mask_t, anchor_area, dist)`. On trigger at frame `k`: composite the warped layer onto frame `k`, save as `warped_k.jpg` in a temp dir, call `generate(warped_k, "clean up warp artifacts; preserve this exact appearance", anchor_frame_path, mask_k)` → new anchor layer at `k` (warp-then-refine §2.6, never from scratch), continue.
  3. Between consecutive anchors, `blend_bidirectional` per frame (past anchor warp + future anchor warp, weights = temporal distance × validity).
  4. Disocclusions (validity 0 inside mask after blend) → `inpaint_service.inpaint_video` scoped to the edit layer region.
  5. `composite(frame, layer, mask_alpha_feathered)` and overwrite frames.
- Consumes: Task 6/7/8 signatures exactly as defined.

- [ ] **Step 1: Write failing test** (fake generator paints a deterministic green box; constant flows; assert one appearance carried, re-anchor invoked on chain cap)

```python
# backend/tests/test_replace_tool.py
import asyncio, cv2, numpy as np, torch
from services import replace_tool

H, W = 48, 64

def _project(tmp_project, n_frames, shift=1):
    for i in range(1, n_frames):
        fwd = np.zeros((2, H, W), np.float16); fwd[0] = shift
        bwd = np.zeros((2, H, W), np.float16); bwd[0] = -shift
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", bwd)
    for t in range(1, n_frames + 1):
        f = np.full((H, W, 3), 90, np.uint8)
        x0 = 10 + (t - 1) * shift
        f[20:30, x0:x0 + 10] = (0, 0, 255)
        cv2.imwrite(str(tmp_project / "frames" / f"frame_{t:04d}.jpg"), f)
        m = np.zeros((H, W), np.uint8); m[20:30, x0:x0 + 10] = 255
        cv2.imwrite(str(tmp_project / "masks" / f"mask_{t:04d}.png"), m)
    return tmp_project

def test_replace_carries_single_appearance(tmp_project):
    p = _project(tmp_project, 6)
    calls = []
    async def fake_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
        calls.append(prompt)
        img = cv2.imread(str(frame_path))
        m = cv2.imread(str(mask_path), 0) if mask_path else None
        img[m > 127] = (0, 255, 0)                     # deterministic green object
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    asyncio.run(replace_tool.apply_replace_range(
        p, list(range(1, 7)), anchor_index=1, prompt="make it green",
        device=torch.device("cpu"), generate=fake_generate))
    assert len(calls) == 1                             # no re-anchor needed on easy clip
    out5 = cv2.imread(str(p / "frames" / "frame_0005.jpg"))
    x = 14 + 4                                          # box center followed the motion
    assert out5[24, x, 1] > 150 and out5[24, x, 2] < 120   # green carried, not red

def test_reanchor_fires_on_chain_cap(tmp_project, monkeypatch):
    monkeypatch.setattr(replace_tool, "CHAIN_CAP", 3)  # shrink 60-frame cap for the test
    p = _project(tmp_project, 8)
    calls = []
    async def fake_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
        calls.append(str(frame_path))
        img = cv2.imread(str(frame_path))
        m = cv2.imread(str(mask_path), 0)
        img[m > 127] = (0, 255, 0)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    asyncio.run(replace_tool.apply_replace_range(
        p, list(range(1, 9)), anchor_index=1, prompt="x",
        device=torch.device("cpu"), generate=fake_generate))
    assert len(calls) >= 2                             # anchor + at least one re-anchor
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# backend/services/replace_tool.py
"""§5.6 replace: generate once → star-warp propagate → warp-then-refine re-anchor.
Never regenerate from scratch (a second independent draw = morphing)."""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from services import mask_service, inpaint_service
from services.propagation_service import (
    EditLayer, make_anchor_layer, star_warp, blend_bidirectional,
    needs_reanchor, composite)

CHAIN_CAP = 60  # frames (~2s @30fps); needs_reanchor also enforces this internally

REFINE_PROMPT = "clean up warp artifacts; preserve this exact appearance"

def _decode(edited_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(edited_bytes, np.uint8), cv2.IMREAD_COLOR)

async def _default_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
    from services import gemini_service
    return await gemini_service.edit_frame_with_reference(
        frame_path, prompt,
        reference_frame_path=reference_frame_path, mask_path=mask_path)

async def apply_replace_range(project_dir: Path, frame_indices, anchor_index: int,
                              prompt: str, device, generate=None) -> None:
    generate = generate or _default_generate
    frames_dir, masks_dir, flows_dir = (project_dir / d for d in ("frames", "masks", "flows"))
    frame_indices = sorted(frame_indices)

    async def gen_layer(idx: int, gen_prompt: str, ref: Path | None,
                        init_frame: Path | None = None) -> EditLayer:
        src = init_frame or frames_dir / f"frame_{idx:04d}.jpg"
        edited = _decode(await generate(src, gen_prompt, reference_frame_path=ref,
                                        mask_path=masks_dir / f"mask_{idx:04d}.png"))
        alpha = mask_service.load_mask_alpha(masks_dir, idx, feather_px=5)
        return make_anchor_layer(edited, alpha)

    anchor_path = frames_dir / f"frame_{anchor_index:04d}.jpg"
    anchors: dict[int, EditLayer] = {anchor_index: await gen_layer(anchor_index, prompt, None)}
    anchor_area = float((mask_service.load_mask_alpha(masks_dir, anchor_index) > 0.5).sum())

    # 1) discover re-anchor points walking outward, warping from the nearest anchor
    for direction in (1, -1):
        cur = anchor_index
        walk = [t for t in frame_indices if (t - anchor_index) * direction > 0]
        for t in (walk if direction == 1 else walk[::-1]):
            dist = abs(t - cur)
            layer = star_warp(anchors[cur], cur, [t], flows_dir, device)[t]
            mask_t = mask_service.load_mask_alpha(masks_dir, t)
            if dist >= CHAIN_CAP or needs_reanchor(layer, mask_t, anchor_area, dist):
                frame_t = cv2.imread(str(frames_dir / f"frame_{t:04d}.jpg"))
                warped_full = composite(frame_t, layer, mask_t)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    cv2.imwrite(tf.name, warped_full)
                    anchors[t] = await gen_layer(t, REFINE_PROMPT, anchor_path,
                                                 init_frame=Path(tf.name))
                cur = t

    # 2) final pass: bidirectional star-warp between bracketing anchors
    anchor_idx_sorted = sorted(anchors)
    for t in frame_indices:
        past = max((a for a in anchor_idx_sorted if a <= t), default=None)
        fut = min((a for a in anchor_idx_sorted if a >= t), default=None)
        if past == t or fut == t:
            layer = anchors[t]
        elif past is not None and fut is not None and past != fut:
            lp = star_warp(anchors[past], past, [t], flows_dir, device)[t]
            lf = star_warp(anchors[fut], fut, [t], flows_dir, device)[t]
            layer = blend_bidirectional(lp, lf, t - past, fut - t)
        else:
            a = past if past is not None else fut
            layer = star_warp(anchors[a], a, [t], flows_dir, device)[t]
        mask_t = mask_service.load_mask_alpha(masks_dir, t, feather_px=5)
        hole = (mask_t > 0.5) & (layer.validity < 0.5)
        frame_t = cv2.imread(str(frames_dir / f"frame_{t:04d}.jpg"))
        out = composite(frame_t, layer, mask_t)
        if hole.any():                                  # disocclusion inside object
            filled = inpaint_service.inpaint_video({t: out}, {t: hole},
                                                   flows_dir, device)[t]
            out = filled
        cv2.imwrite(str(frames_dir / f"frame_{t:04d}.jpg"), out)
```

- [ ] **Step 4: Run tests** — Expected: PASS. Debug direction bugs against Task 6's translation test, not by loosening assertions.

- [ ] **Step 5: Commit**

```bash
git add backend/services/replace_tool.py backend/tests/test_replace_tool.py
git commit -m "feat: replace tool — generate-once star-warp with warp-then-refine re-anchor"
```

---

### Task 12: `background_replace` — soft matte + stabilized plate (§5.5)

**Files:**
- Create: `backend/services/background_tool.py`
- Test: `backend/tests/test_background_tool.py`

**Interfaces:**
- Produces:
  - `soft_matte(frame: np.ndarray, mask_alpha: np.ndarray, radius: int = 8) -> np.ndarray[H,W] float32` — guided filter (`cv2.ximgproc.guidedFilter` if available, else fallback: Gaussian-refined trimap) using the frame as guide so hair/motion-blur edges go soft.
  - `async apply_background_replace_range(project_dir: Path, frame_indices, anchor_index: int, prompt: str, device, generate=None) -> None` — generate the new background ONCE at the anchor (Gemini on the anchor frame with inverted mask; same injectable `generate`), then per frame: estimate global scene translation from the median of the cached fwd flow OUTSIDE the object mask, shift the plate by the accumulated scene motion (camera tracking, spec §5.5.2), composite `out = matte·frame + (1−matte)·plate`.
- Consumes: `flow_service.load_flow`, `mask_service.load_mask_alpha`.

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_background_tool.py
import asyncio, cv2, numpy as np, torch
from services import background_tool

H, W = 48, 64

def test_soft_matte_softens_boundary():
    frame = np.full((H, W, 3), 200, np.uint8)
    mask = np.zeros((H, W), np.float32); mask[10:38, 10:54] = 1.0
    matte = background_tool.soft_matte(frame, mask)
    assert matte[24, 30] > 0.9 and matte[2, 2] < 0.1
    band = ((matte > 0.1) & (matte < 0.9)).sum()
    assert band > 30                                    # soft edge exists

def test_bg_replace_tracks_camera_shift(tmp_project):
    # camera pans right 2px/frame: bg flow = -2 (content moves left)
    for i in (1, 2):
        fwd = np.zeros((2, H, W), np.float16); fwd[0] = -2
        bwd = np.zeros((2, H, W), np.float16); bwd[0] = 2
        np.save(tmp_project / "flows" / f"flow_fwd_{i:04d}.npy", fwd)
        np.save(tmp_project / "flows" / f"flow_bwd_{i:04d}.npy", bwd)
    for t in (1, 2, 3):
        cv2.imwrite(str(tmp_project / "frames" / f"frame_{t:04d}.jpg"),
                    np.full((H, W, 3), 90, np.uint8))
        m = np.zeros((H, W), np.uint8); m[20:30, 28:38] = 255
        cv2.imwrite(str(tmp_project / "masks" / f"mask_{t:04d}.png"), m)
    async def fake_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
        plate = np.zeros((H, W, 3), np.uint8)
        plate[:, ::8] = (255, 255, 255)                 # vertical stripes every 8px
        ok, buf = cv2.imencode(".png", plate)
        return buf.tobytes()
    asyncio.run(background_tool.apply_background_replace_range(
        tmp_project, [1, 2, 3], anchor_index=1, prompt="stripes",
        device=torch.device("cpu"), generate=fake_generate))
    f1 = cv2.imread(str(tmp_project / "frames" / "frame_0001.jpg"))
    f3 = cv2.imread(str(tmp_project / "frames" / "frame_0003.jpg"))
    assert f1[5, 8, 0] > 200                            # stripe at x=8 in frame 1
    assert f3[5, 4, 0] > 200                            # stripes shifted −4px by frame 3
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# backend/services/background_tool.py
"""§5.5 background replace: soft matte + camera-stabilized plate, generated once."""
from pathlib import Path
import cv2
import numpy as np
import torch
from services import flow_service, mask_service

def soft_matte(frame: np.ndarray, mask_alpha: np.ndarray, radius: int = 8) -> np.ndarray:
    guide = frame.astype(np.float32) / 255.0
    src = mask_alpha.astype(np.float32)
    try:
        import cv2.ximgproc as xi
        matte = xi.guidedFilter(guide, src, radius, 1e-4)
    except Exception:
        matte = cv2.GaussianBlur(src, (radius * 2 + 1, radius * 2 + 1), 0)
        core = cv2.erode((src > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), radius // 2)
        matte = np.maximum(matte, core.astype(np.float32))
    return np.clip(matte, 0.0, 1.0)

def _scene_shift(flows_dir: Path, pair_index: int, obj_mask: np.ndarray) -> tuple[float, float]:
    f = flow_service.load_flow(flows_dir, pair_index, "fwd")[0].numpy()
    bg = obj_mask < 0.5
    return float(np.median(f[0][bg])), float(np.median(f[1][bg]))

def _shift_plate(plate: np.ndarray, sx: float, sy: float) -> np.ndarray:
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(plate, M, (plate.shape[1], plate.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)

async def _default_generate(frame_path, prompt, reference_frame_path=None, mask_path=None):
    from services import gemini_service
    return await gemini_service.edit_frame(frame_path, f"replace the background with: {prompt}",
                                           mask_path=mask_path)

async def apply_background_replace_range(project_dir: Path, frame_indices, anchor_index: int,
                                         prompt: str, device, generate=None) -> None:
    generate = generate or _default_generate
    frames_dir, masks_dir, flows_dir = (project_dir / d for d in ("frames", "masks", "flows"))
    frame_indices = sorted(frame_indices)
    raw = await generate(frames_dir / f"frame_{anchor_index:04d}.jpg", prompt,
                         mask_path=masks_dir / f"mask_{anchor_index:04d}.png")
    plate0 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # generate ONCE
    # accumulated scene shift per frame relative to anchor
    shift = {anchor_index: (0.0, 0.0)}
    for t in [i for i in frame_indices if i > anchor_index]:
        px, py = shift[t - 1]
        m = mask_service.load_mask_alpha(masks_dir, t - 1)
        dx, dy = _scene_shift(flows_dir, t - 1, m)
        shift[t] = (px + dx, py + dy)
    for t in sorted((i for i in frame_indices if i < anchor_index), reverse=True):
        px, py = shift[t + 1]
        m = mask_service.load_mask_alpha(masks_dir, t)
        dx, dy = _scene_shift(flows_dir, t, m)
        shift[t] = (px - dx, py - dy)
    for t in frame_indices:
        frame = cv2.imread(str(frames_dir / f"frame_{t:04d}.jpg"))
        matte = soft_matte(frame, mask_service.load_mask_alpha(masks_dir, t))[..., None]
        plate = _shift_plate(plate0, *shift[t]).astype(np.float32)
        out = matte * frame.astype(np.float32) + (1 - matte) * plate
        cv2.imwrite(str(frames_dir / f"frame_{t:04d}.jpg"),
                    np.clip(out, 0, 255).astype(np.uint8))
```

- [ ] **Step 4: Run tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/background_tool.py backend/tests/test_background_tool.py
git commit -m "feat: background_replace with soft matte and camera-stabilized plate"
```

---

### Task 13: Dispatch + main.py rewrite + removals (§6, §8)

**Files:**
- Create: `backend/services/edit_dispatch.py`
- Modify: `backend/main.py` (major surgery — see list)
- Delete: `backend/services/yolo_service.py`, `backend/yolo11n.pt`
- Modify: `backend/requirements.txt` (remove `ultralytics`, `ultralytics-thop`)
- Test: `backend/tests/test_edit_dispatch.py`

**Interfaces:**
- Produces: `async edit_dispatch.run_edit_rule(project_id: str, rule: EditRule, progress_cb=None) -> None` — the §6 dispatch table:

```
recolor, blur_region, color_pop, glow -> per-frame local_edit_service (§5.1)
resize                                -> object_tools.apply_resize_range (§5.2)
delete                                -> object_tools.apply_delete_range (§5.3)
move                                  -> object_tools.apply_move_range (§5.4)
bg_replace                            -> background_tool.apply_background_replace_range (§5.5)
replace                               -> propagation engine via replace_tool (§5.6);
                                         Backend B if config.USE_SYNTH (Task 15 stub)
```

  It ensures flows exist first (`flow_service.compute_flows` — cached, cheap on re-edit), ensures SAM 2 masks cover the range (`sam2_service.propagate_masks` + `mask_service.stabilize_masks`), reports progress via `progress_cb(done, total)`.
- Modify `EditRule` (main.py:291): keep existing fields, add `dx: Optional[int] = None`, `dy: Optional[int] = None` (move), `backend: Optional[str] = None` (Backend A/B override). Prune the dead `edit_type` comment list to the §6 set.
- `main.py` `_background_edit` (:387) becomes a thin loop: for each rule → `await edit_dispatch.run_edit_rule(...)`, keeping the existing undo-snapshot behavior (`_save_edited_frame`'s `.orig` backups) and status updates.
- **Deletions in `main.py`:** `/ai/edit/preview|accept|reject|retry` + `AIPreviewRequest/AIAcceptRequest/AIRejectRequest/AIRetryRequest` + `_background_ai_edit` (:1195) + `/preview/{...}` route; `/edit/refine` + `RefineRequest` (:630); `_background_propagate_changes` (:994) and `/edit/propagate` + `PropagateRequest/ChangeLogEntry` (the changelog propagation path — the frontend moves to plain `/edit` rules in Task 16); commented `/detect` block (:170); `storage_service` import if unused. `rife_service`/`film_service` imports stay only if still referenced by slow-mo code; RIFE must not be reachable from any edit path.

- [ ] **Step 1: Write failing test** (dispatch routes to the right service; deterministic path touches every frame)

```python
# backend/tests/test_edit_dispatch.py
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from services import edit_dispatch

def _rule(**kw):
    from main import EditRule
    return EditRule(**{"edit_type": "recolor", "start_frame": 1, "end_frame": 3, **kw})

def test_deterministic_rule_hits_every_frame(tmp_project):
    with patch.object(edit_dispatch, "_ensure_flows"), \
         patch.object(edit_dispatch, "_ensure_masks"), \
         patch.object(edit_dispatch, "_project_dir", return_value=tmp_project), \
         patch("services.local_edit_service.apply_recolor") as rec:
        asyncio.run(edit_dispatch.run_edit_rule("pid", _rule(color="00FF00")))
        assert rec.call_count == 3

def test_replace_routes_to_propagation_engine(tmp_project):
    with patch.object(edit_dispatch, "_ensure_flows"), \
         patch.object(edit_dispatch, "_ensure_masks"), \
         patch.object(edit_dispatch, "_project_dir", return_value=tmp_project), \
         patch("services.replace_tool.apply_replace_range", new_callable=AsyncMock) as rep:
        asyncio.run(edit_dispatch.run_edit_rule("pid", _rule(edit_type="replace", prompt="a dog")))
        rep.assert_awaited_once()
        assert rep.await_args.kwargs.get("prompt") == "a dog" or "a dog" in rep.await_args.args
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `edit_dispatch.py`**

```python
# backend/services/edit_dispatch.py
"""§6 dispatch: deterministic per-frame vs generative propagation."""
from pathlib import Path
import asyncio
from services import (config, flow_service, mask_service, local_edit_service,
                      object_tools, background_tool, replace_tool, project_manager)

DETERMINISTIC = {"recolor", "blur_region", "color_pop", "glow"}

def _project_dir(project_id: str) -> Path:
    return project_manager.get_project_dir(project_id)

def _ensure_flows(project_dir: Path):
    flow_service.compute_flows(project_dir / "frames", project_dir / "flows",
                               device=config.get_device())

def _ensure_masks(project_dir: Path, start: int, end: int):
    masks_dir = project_dir / "masks"
    missing = [t for t in range(start, end + 1)
               if not (masks_dir / f"mask_{t:04d}.png").exists()]
    if missing:
        from services import sam2_service
        sam2_service.propagate_masks(project_dir / "frames", masks_dir, missing)
        mask_service.stabilize_masks(masks_dir)

async def run_edit_rule(project_id: str, rule, progress_cb=None) -> None:
    project_dir = _project_dir(project_id)
    device = config.get_device()
    indices = list(range(rule.start_frame, rule.end_frame + 1))
    status = project_manager.get_status(project_id)
    anchor = status.get("anchor_frame") or rule.start_frame
    anchor = min(max(anchor, rule.start_frame), rule.end_frame)

    if rule.edit_type in DETERMINISTIC:
        _ensure_masks(project_dir, rule.start_frame, rule.end_frame)
        loop = asyncio.get_event_loop()
        for n, t in enumerate(indices, 1):
            fp = project_dir / "frames" / f"frame_{t:04d}.jpg"
            mp = project_dir / "masks" / f"mask_{t:04d}.png"
            if rule.edit_type == "recolor":
                await loop.run_in_executor(None, local_edit_service.apply_recolor, fp, mp, rule.color or "FF0000")
            elif rule.edit_type == "blur_region":
                await loop.run_in_executor(None, local_edit_service.apply_blur_region, fp, mp, rule.blur_strength or 10)
            elif rule.edit_type == "color_pop":
                await loop.run_in_executor(None, local_edit_service.apply_color_pop, fp, mp)
            elif rule.edit_type == "glow":
                await loop.run_in_executor(None, local_edit_service.apply_glow, fp, mp)
            if progress_cb:
                progress_cb(n, len(indices))
        return

    _ensure_flows(project_dir)                      # generative + transport tools need flow
    _ensure_masks(project_dir, rule.start_frame, rule.end_frame)

    if rule.edit_type == "delete":
        object_tools.apply_delete_range(project_dir, indices, device)
    elif rule.edit_type == "resize":
        object_tools.apply_resize_range(project_dir, indices, rule.scale or 1.5, device)
    elif rule.edit_type == "move":
        offsets = {t: (rule.dx or 0, rule.dy or 0) for t in indices}
        object_tools.apply_move_range(project_dir, indices, offsets, device)
    elif rule.edit_type == "bg_replace":
        await background_tool.apply_background_replace_range(
            project_dir, indices, anchor, rule.prompt or "", device)
    elif rule.edit_type == "replace":
        if config.USE_SYNTH:
            from services import synth_propagation_service
            await synth_propagation_service.apply_replace_range(
                project_dir, indices, anchor, rule.prompt or "", device)
        else:
            await replace_tool.apply_replace_range(
                project_dir, indices, anchor, rule.prompt or "", device)
    else:
        raise ValueError(f"Unknown edit_type: {rule.edit_type}")
    if progress_cb:
        progress_cb(len(indices), len(indices))
```

- [ ] **Step 4: main.py surgery** — perform the deletions listed in Interfaces (grep for each symbol; the line numbers shift as you delete — anchor on names, not numbers). Rewrite `_background_edit` to snapshot originals (existing behavior), then `for rule in edit_rules: await edit_dispatch.run_edit_rule(project_id, rule, progress_cb)` with status updates `edit_status/edit_progress`. Update `EditRule` with `dx`, `dy`, `backend`. Delete `services/yolo_service.py`, `yolo11n.pt`; remove `ultralytics`/`ultralytics-thop` from `requirements.txt`.

- [ ] **Step 5: Run tests + boot smoke** —
`./venv/bin/python -m pytest tests/ -v` → all PASS.
`./venv/bin/python -c "import main"` → imports cleanly (catches deleted-symbol references).

- [ ] **Step 6: Commit**

```bash
git add -A backend
git commit -m "feat: unified edit dispatch; remove AI-chat/refine/YOLO paths and RIFE from edit pipeline"
```

---

### Task 14: Backend B stub (`synth_propagation_service`) behind flags

**Files:**
- Create: `backend/services/synth_propagation_service.py`
- Test: `backend/tests/test_synth_stub.py`

**Interfaces:**
- Produces: `async apply_replace_range(project_dir, frame_indices, anchor_index, prompt, device, generate=None)` — same signature as `replace_tool.apply_replace_range` (§3.3 interface parity). Behavior today: if `config.USE_CUDA_KERNEL` → raise `NotImplementedError("Doc 2 Track B kernel not built")`; else log a fallback warning and delegate verbatim to `replace_tool.apply_replace_range` (Backend A is always the fallback — Doc 2 §0).

- [ ] **Step 1: Write failing test**

```python
# backend/tests/test_synth_stub.py
import asyncio, torch
from unittest.mock import patch, AsyncMock
from services import synth_propagation_service

def test_stub_falls_back_to_backend_a(tmp_project):
    with patch("services.replace_tool.apply_replace_range", new_callable=AsyncMock) as a:
        asyncio.run(synth_propagation_service.apply_replace_range(
            tmp_project, [1, 2], 1, "x", torch.device("cpu")))
        a.assert_awaited_once()
```

- [ ] **Step 2: Verify failure, implement, verify pass**

```python
# backend/services/synth_propagation_service.py
"""Backend B (Doc 1 §3 / Doc 2 Track B). Stub: Backend A is always the fallback."""
from services import config, replace_tool

async def apply_replace_range(project_dir, frame_indices, anchor_index, prompt,
                              device, generate=None):
    if config.USE_CUDA_KERNEL:
        raise NotImplementedError("Doc 2 Track B kernel not built")
    print("[synth] Backend B not available — falling back to Backend A (flow warp)")
    return await replace_tool.apply_replace_range(
        project_dir, frame_indices, anchor_index, prompt, device, generate=generate)
```

- [ ] **Step 3: Commit**

```bash
git add backend/services/synth_propagation_service.py backend/tests/test_synth_stub.py
git commit -m "feat: Backend B stub with Backend A fallback (Doc 2 off critical path)"
```

---

### Task 15: Frontend — toolbar rework, move tool, removals

**Files:**
- Modify: `frontend/src/components/editor/EditToolbar.tsx` (tool list :20–:49)
- Modify: `frontend/src/hooks/useEditorState.ts` (drop `/ai/edit` + `/edit/refine` + `/edit/propagate` calls; add move offset submission)
- Modify: `frontend/src/app/editor/[projectId]/page.tsx` (remove `AIChatPane` usage)
- Delete: `frontend/src/components/editor/AIChatPane.tsx`, `frontend/src/components/editor/AIProgressOverlay.tsx`
- Modify: `frontend/src/components/editor/BoundingBox.tsx` (draggable → emits `{dx, dy}`)

**Interfaces:**
- Consumes: backend `EditRule` now accepts `edit_type ∈ {recolor, blur_region, color_pop, glow, resize, delete, move, bg_replace, replace}` plus `dx`/`dy` for move.
- Produces: toolbar tool union type matching that list exactly.

- [ ] **Step 1: Rework the tool list in `EditToolbar.tsx`**

Replace the `ToolId` union (:20–:28) and `TOOLS` array (:41–:49):

```tsx
type ToolId =
  | "recolor"
  | "resize"
  | "delete"
  | "blur_region"
  | "move"
  | "color_pop"
  | "glow"
  | "replace"
  | "bg_replace";

const TOOLS: Tool[] = [
  { id: "delete", icon: Trash2, label: "Remove", category: "object" },
  { id: "recolor", icon: Palette, label: "Recolor", needsColor: true, category: "object" },
  { id: "resize", icon: Maximize2, label: "Resize", needsScale: true, category: "object" },
  { id: "blur_region", icon: EyeOff, label: "Blur", category: "object" },
  { id: "move", icon: Move, label: "Move", category: "object" },
  { id: "color_pop", icon: Droplet, label: "Color Pop", category: "object" },
  { id: "glow", icon: Sun, label: "Glow", category: "object" },
  { id: "replace", icon: RefreshCw, label: "Replace", needsPrompt: true, category: "object" },
  { id: "bg_replace", icon: ImagePlus, label: "Replace BG", needsPrompt: true, category: "frame" },
];
```

Import `Move`, `Droplet`, `Sun`, `RefreshCw` from `lucide-react`. Remove `enhance`/`upscale`/`restore` entries, their icon imports, and any `switch` cases referencing them in this file and `useEditorState.ts`.

- [ ] **Step 2: Move-tool offsets** — in `BoundingBox.tsx`, when the active tool is `move`, make the box draggable (pointer events on the overlay div; track `dragStart`/`current` in local state) and call an `onMove?(dx: number, dy: number)` prop on pointer-up with canvas-space deltas scaled to frame pixels (`dx * frameWidth / canvasWidth`). In `useEditorState.ts`, store `moveOffset` and include `dx`, `dy` in the `EditRule` payload when `edit_type === "move"`.

- [ ] **Step 3: Removals** — delete `AIChatPane.tsx`, `AIProgressOverlay.tsx`; strip their imports/JSX from `page.tsx` and state from `useEditorState.ts`; delete all fetches to `/ai/edit/*`, `/edit/refine`, `/edit/propagate` (grep `src/` for each path — `changeLogStore.ts` likely goes too if only the propagate path used it).

- [ ] **Step 4: Verify** —
Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: type-check and build succeed with zero references to removed endpoints (`grep -rn "ai/edit\|refine\|propagate" src/` → empty).

- [ ] **Step 5: Commit**

```bash
git add -A frontend
git commit -m "feat: toolbar rework (move/color-pop/glow/replace), remove AI chat and enhance family"
```

---

### Task 16: Modal deploy config

**Files:**
- Create: `backend/modal_app.py`
- Create: `backend/.env.example` (document required secrets)

**Interfaces:**
- Produces: `modal_app.py` deployable with `modal deploy modal_app.py`; serves the existing FastAPI `app` unchanged; `modal.Volume` named `frameshift-projects` mounted at the projects dir; SAM 2 + RAFT warmed in a container-enter hook with memory snapshots.
- Consumes: `main.app` (FastAPI instance), env secrets `GEMINI_API_KEY`, `CLOUDINARY_*`.

- [ ] **Step 1: Write `modal_app.py`**

```python
# backend/modal_app.py
"""Modal deploy (Doc 1 §9). No custom native builds on this path —
Backend A is torch-native; Doc 2 kernels are dev-only."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0",
        extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("services", "/root/services")
    .add_local_file("main.py", "/root/main.py")
    .add_local_dir("checkpoints", "/root/checkpoints")
)

app = modal.App("frameshift", image=image)
volume = modal.Volume.from_name("frameshift-projects", create_if_missing=True)

@app.cls(
    gpu="A10G",
    volumes={"/root/projects": volume},
    secrets=[modal.Secret.from_name("frameshift-secrets")],  # GEMINI_API_KEY, CLOUDINARY_*
    scaledown_window=120,
    enable_memory_snapshot=True,
)
class Server:
    @modal.enter(snap=True)
    def warm(self):
        from services import sam2_service, flow_service
        import torch
        sam2_service.get_image_predictor()
        flow_service._build_raft(torch.device("cuda"))

    @modal.asgi_app()
    def fastapi_app(self):
        from main import app as fastapi
        from fastapi.middleware.cors import CORSMiddleware
        fastapi.add_middleware(
            CORSMiddleware,
            allow_origins=["https://frameshift.vercel.app"],  # pin to real Vercel domain
            allow_methods=["*"], allow_headers=["*"])
        return fastapi
```

Add a simple per-IP token bucket as FastAPI middleware in `main.py` (dict of `ip -> (tokens, last_refill)`, 30 req/min, refill 0.5/s; return 429 when empty) guarded by `os.getenv("RATE_LIMIT", "0") == "1"` so local dev is unaffected.

- [ ] **Step 2: Local validation** (no Modal account needed): `./venv/bin/python -c "import ast; ast.parse(open('modal_app.py').read())"` and `./venv/bin/python -m pytest tests/ -v` still green. Note in commit message that actual `modal deploy` needs `modal token new` (user credentials).

- [ ] **Step 3: Commit**

```bash
git add backend/modal_app.py backend/.env.example backend/main.py
git commit -m "feat: Modal ASGI deploy config with warm snapshots (deploy requires modal token)"
```

---

### Task 17: End-to-end acceptance harness (§7, §11 — "play it and watch it")

**Files:**
- Create: `backend/scripts/acceptance.py`

**Interfaces:**
- Produces: `./venv/bin/python scripts/acceptance.py <video.mp4> --tool <tool> [--prompt ...] [--color ...] [--scale ...] [--dx --dy]` — runs the REAL pipeline headless: upload→extract→segment (center click of frame 1)→edit→render, writes `output/acceptance_<tool>.mp4`, prints per-stage wall time and the upload→editor-ready time against the <10s target. One clip per tool = the §11 acceptance table, watched by a human.

- [ ] **Step 1: Write the script** (call service functions directly, not HTTP; mirror `_background_extract`/`_background_segment_and_propagate`/`edit_dispatch.run_edit_rule`/`encode_video` in sequence with `time.perf_counter()` between stages).

- [ ] **Step 2: Run the full matrix on a real 5–10s clip** (needs `GEMINI_API_KEY` in `backend/.env` for replace/bg_replace):

```bash
cd backend
for t in recolor blur_region color_pop glow resize delete move; do
  ./venv/bin/python scripts/acceptance.py input/test.mp4 --tool $t || echo "FAIL: $t"
done
./venv/bin/python scripts/acceptance.py input/test.mp4 --tool replace --prompt "a golden retriever"
./venv/bin/python scripts/acceptance.py input/test.mp4 --tool bg_replace --prompt "sunset beach"
```

Expected: every command exits 0 and produces a playable mp4. **Watch each clip** against the §11 table (no morph, no flicker, no seam crawl, no cut line). Fix what fails before calling the pipeline done — the unit is the clip.

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/acceptance.py
git commit -m "feat: headless per-tool acceptance harness"
```

---

## Self-Review Notes

- **Spec coverage:** §2 → Tasks 3–7; §3 → Task 14 (stub, per scope); §4 → Task 8; §5.1 → Task 9; §5.2–5.4 → Task 10; §5.5 → Task 12; §5.6 → Task 11; §6 → Task 13; §7 → Tasks 2, 17; §8 → Tasks 9, 13, 15; §9 → Task 16; §10/§11 → Task 17. Doc 2 explicitly out of scope beyond the flag plumbing (Tasks 1, 14).
- **Full-res render (§7):** shipped as 720p output in v1 (spec allows: "720p flows/masks upsample cleanly" is listed as the v2 path). Documented compromise, revisit post-acceptance.
- **Type consistency:** `EditLayer`, `star_warp`, `blend_bidirectional`, `needs_reanchor`, `inpaint_video`, `apply_replace_range` signatures are defined once (Tasks 6–8, 11) and consumed by name in Tasks 10–14.
- **Known risk:** flow-direction bookkeeping in `_compose_chain` — covered by translation tests in Tasks 3, 6, 11; debug against those, never by loosening tolerances.
