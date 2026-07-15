# FrameShift on Modal CUDA

This is the immediate production configuration: one NVIDIA GPU runs SAM2,
RAFT, and torch-native Backend A. The API stays responsive to status polling,
but a process-wide GPU queue prevents SAM2 and RAFT from executing together.

The resulting order is:

```text
upload -> extract -> keyframe SAM2 -> user confirms
       -> SAM2 tracking -> lazy batched RAFT -> edit propagation -> encode
```

RAFT is not launched during extraction. It starts only after mask tracking is
complete and an edit needs flow. Flow files and project artifacts are cached in
the `frameshift-projects` Modal Volume.

## 1. Install and authenticate Modal

Run these commands from `backend/`:

```bash
venv/bin/pip install modal
venv/bin/modal setup
```

If the virtual environment is not active, use the explicit paths above. With
an activated environment, `modal setup` is sufficient.

## 2. Create the backend secret

```bash
cp .env.modal.example .env.modal
```

Edit `.env.modal`:

```dotenv
GEMINI_API_KEY=your-key
FRONTEND_ORIGIN=https://your-frontend.vercel.app
AUTH0_DOMAIN=
AUTH0_AUDIENCE=
```

`FRONTEND_ORIGIN` must exactly match the deployed frontend origin and should
not have a trailing slash. Auth0 values are optional unless the frontend sends
Auth0 access tokens.

Create or update the Modal secret:

```bash
venv/bin/modal secret create frameshift-secrets --from-dotenv .env.modal --force
```

## 3. Deploy

L40S is the default and the recommended first performance benchmark:

```bash
FRAMESHIFT_MODAL_GPU=L40S venv/bin/modal deploy modal_app.py
```

For the lower-cost test deployment:

```bash
FRAMESHIFT_MODAL_GPU=L4 venv/bin/modal deploy modal_app.py
```

`A10` and `H100` are also accepted. The selected GPU is read when the Modal
app is deployed, so changing GPU type requires another deploy.

The deploy command prints an HTTPS endpoint ending in `modal.run`. Verify it:

```bash
curl https://YOUR-MODAL-ENDPOINT.modal.run/health
```

The response should report `"compute_device":"cuda"`.

## 4. Connect the frontend

Set this build-time environment variable in Vercel:

```dotenv
NEXT_PUBLIC_API_URL=https://YOUR-MODAL-ENDPOINT.modal.run
```

Then redeploy the frontend. Because it is a `NEXT_PUBLIC_` variable, changing
it without rebuilding the frontend will not update the browser bundle.

## Runtime choices

- `max_containers=1` gives one project-wide GPU queue and avoids cross-container
  Volume consistency problems during the first deployment.
- Concurrent HTTP inputs keep frame/status requests responsive while a GPU job
  runs; the application lock still serializes SAM2 and RAFT.
- `FRAMESHIFT_PRECOMPUTE_FLOWS=0` prevents extraction from competing with
  segmentation.
- RAFT batches 2 adjacent frame pairs per call on L4/A10 and 4 on L40S/H100.
  Override with `FRAMESHIFT_RAFT_BATCH_SIZE` before deploy if GPU memory is
  constrained.
- `FRAMESHIFT_USE_SYNTH=0` and `FRAMESHIFT_USE_CUDA_KERNEL=0` select Backend A.
- GPU memory snapshots retain the warmed SAM2 and RAFT models to reduce cold
  starts. This Modal capability is currently marked experimental.
- `scaledown_window=180` keeps the container warm for three idle minutes. Raise
  it for lower cold-start latency or lower it to reduce idle GPU cost.

This first configuration deliberately attaches the GPU to the ASGI server. It
is the smallest migration from the local backend. After the CUDA benchmark is
green, the cost-optimized architecture is a CPU API plus a separate GPU worker
queue, so uploads, polling, and FFmpeg-only work do not hold a billed GPU.
