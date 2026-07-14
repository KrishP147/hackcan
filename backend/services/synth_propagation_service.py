"""Backend B — Ezsynth-inspired guided synthesis (Doc 1 §3 / Doc 2 Track B).

Interface parity with replace_tool.apply_replace_range (§3.3). Today this is a
stub: Backend A is ALWAYS the fallback (Doc 2 §0) — deployment never depends
on the CUDA kernel compiling."""
from services import config, replace_tool


async def apply_replace_range(project_dir, frame_indices, anchor_index, prompt,
                              device, generate=None):
    if config.USE_CUDA_KERNEL:
        raise NotImplementedError("Doc 2 Track B kernel not built")
    print("[synth] Backend B not available — falling back to Backend A (flow warp)")
    return await replace_tool.apply_replace_range(
        project_dir, frame_indices, anchor_index, prompt, device, generate=generate)
