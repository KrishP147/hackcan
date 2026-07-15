"""§6 dispatch: deterministic per-frame tools vs generative propagation.

recolor, blur_region, color_pop, glow -> per-frame local_edit_service (§5.1)
resize                                -> object_tools.apply_resize_range (§5.2)
delete                                -> object_tools.apply_delete_range (§5.3)
move                                  -> object_tools.apply_move_range (§5.4)
bg_replace                            -> background_tool (§5.5)
replace                               -> propagation engine (§5.6); Backend B if USE_SYNTH
"""
import asyncio
from pathlib import Path

import numpy as np

from services import (background_tool, config, flow_service, local_edit_service,
                      mask_service, object_tools, project_manager, replace_tool)

DETERMINISTIC = {"recolor", "blur_region", "color_pop", "glow"}


class EditCancelled(RuntimeError):
    """Raised between frames when the user cancels an edit job."""


def _project_dir(project_id: str) -> Path:
    return project_manager.get_project_dir(project_id)


def _ensure_flows(project_dir: Path):
    """RAFT flow on original footage — cached once per project, cheap on re-edit."""
    flow_service.compute_flows(project_dir / "frames", project_dir / "flows",
                               device=config.get_device())


def _ensure_masks(project_id: str, project_dir: Path, start: int, end: int,
                  cancel_check=None):
    """SAM 2-propagate masks over any frames in range that lack one, then stabilize."""
    masks_dir = project_dir / "masks"
    missing = [t for t in range(start, end + 1)
               if not (masks_dir / f"mask_{t:04d}.png").exists()]
    if not missing:
        return
    status = project_manager.get_status(project_id)
    anchor = status.get("anchor_frame")
    anchor_mask_path = masks_dir / f"mask_{anchor:04d}.png" if anchor else None
    if not anchor or not anchor_mask_path.exists():
        raise RuntimeError("No segmentation anchor — click an object first")
    from PIL import Image
    anchor_mask = (np.array(Image.open(anchor_mask_path).convert("L")) > 128)
    from services import sam2_service
    sam2_service.propagate_masks(
        project_dir / "frames", anchor, anchor_mask, masks_dir,
        click_x=status.get("click_x"), click_y=status.get("click_y"),
        frame_step=config.get_mask_frame_step(), cancel_check=cancel_check)
    mask_service.stabilize_masks(masks_dir)
    project_manager.update_status(
        project_id,
        mask_count=len(list(masks_dir.glob("mask_*.png"))),
    )


async def run_edit_rule(project_id: str, rule, progress_cb=None, cancel_check=None) -> None:
    project_dir = _project_dir(project_id)
    device = config.get_device()
    loop = asyncio.get_running_loop()
    indices = list(range(rule.start_frame, rule.end_frame + 1))
    status = project_manager.get_status(project_id)
    anchor = status.get("anchor_frame") or rule.start_frame
    anchor = min(max(anchor, rule.start_frame), rule.end_frame)

    if rule.edit_type in DETERMINISTIC:
        await loop.run_in_executor(
            None, _ensure_masks, project_id, project_dir,
            rule.start_frame, rule.end_frame, cancel_check,
        )
        for n, t in enumerate(indices, 1):
            if cancel_check and cancel_check():
                raise EditCancelled()
            fp = project_dir / "frames" / f"frame_{t:04d}.jpg"
            mp = project_dir / "masks" / f"mask_{t:04d}.png"
            if not fp.exists():
                continue
            if rule.edit_type == "recolor":
                await loop.run_in_executor(
                    None, local_edit_service.apply_recolor, fp, mp, rule.color or "FF0000")
            elif rule.edit_type == "blur_region":
                await loop.run_in_executor(
                    None, local_edit_service.apply_blur_region, fp, mp,
                    rule.blur_strength or 10)
            elif rule.edit_type == "color_pop":
                await loop.run_in_executor(None, local_edit_service.apply_color_pop, fp, mp)
            elif rule.edit_type == "glow":
                await loop.run_in_executor(None, local_edit_service.apply_glow, fp, mp)
            if progress_cb:
                progress_cb(n, len(indices))
        return

    await loop.run_in_executor(None, _ensure_flows, project_dir)
    await loop.run_in_executor(
        None, _ensure_masks, project_id, project_dir,
        rule.start_frame, rule.end_frame, cancel_check,
    )

    if rule.edit_type == "delete":
        await loop.run_in_executor(
            None, object_tools.apply_delete_range, project_dir, indices, device)
    elif rule.edit_type == "resize":
        await loop.run_in_executor(
            None, object_tools.apply_resize_range,
            project_dir, indices, rule.scale or 1.5, device)
    elif rule.edit_type == "move":
        offsets = {t: (rule.dx or 0, rule.dy or 0) for t in indices}
        await loop.run_in_executor(
            None, object_tools.apply_move_range,
            project_dir, indices, offsets, device)
    elif rule.edit_type == "bg_replace":
        await background_tool.apply_background_replace_range(
            project_dir, indices, anchor, rule.prompt or "", device)
    elif rule.edit_type == "replace":
        if config.USE_SYNTH or (getattr(rule, "backend", None) == "B"):
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
