# FrameShift guided-synthesis CUDA extension

This is Document 2, Track B: guided jump-flood PatchMatch plus overlapping
patch voting. It is optional and never required by the production deployment.

Build on a CUDA development machine from `backend/`:

```bash
venv/bin/python ebsynth_synth/setup.py build_ext --inplace
```

Enable it with:

```bash
FRAMESHIFT_USE_SYNTH=1 FRAMESHIFT_USE_CUDA_KERNEL=1 \
  venv/bin/uvicorn main:app --reload
```

If import or ABI loading fails, Backend B automatically uses the portable
PyTorch PatchMatch reference. Backend A remains available by setting
`FRAMESHIFT_USE_SYNTH=0`.
