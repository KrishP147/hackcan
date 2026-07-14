"""Headless per-tool acceptance harness (Doc 1 §7 target, §11 table).

Runs the REAL pipeline (no HTTP): upload -> extract -> segment (center click)
-> edit -> render, printing per-stage wall time. The unit of acceptance is the
rendered clip — watch it.

Usage:
  ./venv/bin/python scripts/acceptance.py <video.mp4> --tool recolor --color 00FF00
  ./venv/bin/python scripts/acceptance.py <video.mp4> --tool replace --prompt "a dog"
  ./venv/bin/python scripts/acceptance.py <video.mp4> --tool move --dx 80 --dy 0 \
      --reuse <project_id>   # clone pristine frames/masks/flows from a previous run

--reuse clones the source project's PRISTINE artifacts (frames_orig snapshot,
masks, flows) into a new project, so runs never compound edits on edited frames.

Output: output/acceptance_<tool>.mp4 + the project id for --reuse.
"""
import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import config, edit_dispatch, ffmpeg_service, mask_service, project_manager  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--color", default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--dx", type=int, default=None)
    ap.add_argument("--dy", type=int, default=None)
    ap.add_argument("--reuse", default=None, help="existing project id (skip extract/segment)")
    ap.add_argument("--max-seconds", type=float, default=5.0,
                    help="trim the input clip to this length first")
    args = ap.parse_args()

    stages: list[tuple[str, float]] = []

    def stage(name):
        stages.append((name, time.perf_counter()))
        print(f"[{time.strftime('%H:%M:%S')}] {name}...")

    if args.reuse:
        src_dir = project_manager.get_project_dir(args.reuse)
        src_frames = src_dir / "frames_orig"
        assert src_frames.exists(), \
            f"project {args.reuse} has no frames_orig snapshot (re-run without --reuse)"
        stage("reuse (clone pristine)")
        project = project_manager.create_project()
        project_id = project["project_id"]
        project_dir = project_manager.get_project_dir(project_id)
        frames_dir = project_dir / "frames"
        shutil.copytree(src_frames, frames_dir, dirs_exist_ok=True)
        if (src_dir / "masks").exists():
            shutil.copytree(src_dir / "masks", project_dir / "masks", dirs_exist_ok=True)
        if (src_dir / "flows").exists():
            # flows are never rewritten (compute_flows skips existing) — hardlink,
            # they're ~2GB per project and would fill the temp volume if copied
            import os
            try:
                shutil.copytree(src_dir / "flows", project_dir / "flows",
                                copy_function=os.link, dirs_exist_ok=True)
            except OSError:
                shutil.copytree(src_dir / "flows", project_dir / "flows",
                                dirs_exist_ok=True)
        src_status = project_manager.get_status(args.reuse)
        project_manager.update_status(
            project_id, status="ready",
            **{k: src_status[k] for k in
               ("frame_count", "fps", "anchor_frame", "click_x", "click_y", "mask_count")
               if src_status.get(k) is not None})
    else:
        stage("upload")
        project = project_manager.create_project()
        project_id = project["project_id"]
        project_dir = project_manager.get_project_dir(project_id)
        video_path = project_dir / "original.mp4"
        # trim to max-seconds so acceptance runs stay fast
        import subprocess
        subprocess.run([ffmpeg_service._FFMPEG, "-y", "-i", str(args.video),
                        "-t", str(args.max_seconds), "-c", "copy", str(video_path)],
                       check=True, capture_output=True)

        stage("extract")
        frames_dir = project_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        count, fps = ffmpeg_service.extract_frames(video_path, frames_dir)
        project_manager.update_status(project_id, status="ready", frame_count=count, fps=fps)
        print(f"    {count} frames @ {fps:.2f}fps")
        # pristine snapshot so --reuse runs never see edited frames
        shutil.copytree(frames_dir, project_dir / "frames_orig", dirs_exist_ok=True)

        stage("segment (center click)")
        import cv2
        from services import sam2_service
        first = cv2.imread(str(frames_dir / "frame_0001.jpg"))
        h, w = first.shape[:2]
        cx, cy = w // 2, h // 2
        mask = sam2_service.segment_frame(frames_dir / "frame_0001.jpg", cx, cy)
        masks_dir = project_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        from PIL import Image
        Image.fromarray((mask.astype("uint8")) * 255).save(masks_dir / "mask_0001.png")
        mask_service.condition_single(masks_dir / "mask_0001.png")
        project_manager.update_status(project_id, anchor_frame=1, click_x=cx, click_y=cy,
                                      mask_count=1)

    stage(f"edit ({args.tool})")
    from main import EditRule  # after path setup
    status = project_manager.get_status(project_id)
    n_frames = status.get("frame_count") or len(list(frames_dir.glob("frame_*.jpg")))
    rule = EditRule(edit_type=args.tool, start_frame=1, end_frame=n_frames,
                    prompt=args.prompt, color=args.color, scale=args.scale,
                    dx=args.dx, dy=args.dy)
    asyncio.run(edit_dispatch.run_edit_rule(project_id, rule))

    stage("render")
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"acceptance_{args.tool}.mp4"
    ffmpeg_service.encode_video(frames_dir, out_path, fps=status.get("fps") or 30)

    stages.append(("done", time.perf_counter()))
    if args.reuse:
        # the clone served its purpose once the mp4 is rendered — free the disk
        shutil.rmtree(project_dir, ignore_errors=True)
        project_id = args.reuse
    print(f"\n=== {args.tool} on project {project_id} ===")
    for (name, t0), (_, t1) in zip(stages, stages[1:]):
        print(f"  {name:28s} {t1 - t0:7.1f}s")
    upload_to_ready = next((t for n, t in stages if n.startswith("edit")), None)
    if not args.reuse and upload_to_ready:
        total_pre = upload_to_ready - stages[0][1]
        verdict = "OK" if total_pre < 10 else "OVER TARGET"
        print(f"  upload->editor-ready         {total_pre:7.1f}s  [{verdict}, target <10s]")
    print(f"  output: {out_path}")
    print(f"  reuse:  --reuse {project_id}")


if __name__ == "__main__":
    main()
