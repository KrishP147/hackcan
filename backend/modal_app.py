"""Single-GPU Modal deployment for FrameShift's production Backend A.

The ASGI server, SAM2, and RAFT share one CUDA container. Heavy neural stages
are serialized by services.config.gpu_job while concurrent HTTP inputs keep
status polling responsive.

Deploy from backend/:
    modal setup
    modal secret create frameshift-secrets --from-dotenv .env.modal
    FRAMESHIFT_MODAL_GPU=L40S modal deploy modal_app.py
"""
import os

import modal

GPU_TYPE = os.getenv("FRAMESHIFT_MODAL_GPU", "L40S").upper()
SUPPORTED_GPUS = {"L4", "A10", "L40S", "H100"}
if GPU_TYPE not in SUPPORTED_GPUS:
    raise ValueError(
        f"FRAMESHIFT_MODAL_GPU must be one of {sorted(SUPPORTED_GPUS)}, got {GPU_TYPE!r}"
    )

RAFT_BATCH_SIZE = os.getenv(
    "FRAMESHIFT_RAFT_BATCH_SIZE",
    "4" if GPU_TYPE in {"L40S", "H100"} else "2",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    # Install the CUDA wheel before the general requirements. The same pinned
    # versions in requirements.txt are then already satisfied and cannot
    # replace this with a CPU wheel.
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install_from_requirements("requirements.txt")
    .env({
        "PYTHONPATH": "/root",
        "FRAMESHIFT_PROJECTS_DIR": "/root/projects",
        "FRAMESHIFT_PRECOMPUTE_FLOWS": "0",
        "FRAMESHIFT_RAFT_BATCH_SIZE": RAFT_BATCH_SIZE,
        "FRAMESHIFT_USE_SYNTH": "0",
        "FRAMESHIFT_USE_CUDA_KERNEL": "0",
        "RATE_LIMIT": "0",
    })
    .add_local_dir("services", "/root/services")
    .add_local_dir("rife_vendor", "/root/rife_vendor")
    .add_local_file("main.py", "/root/main.py")
    .add_local_dir("checkpoints", "/root/checkpoints")
)

app = modal.App("frameshift")
volume = modal.Volume.from_name("frameshift-projects", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/projects": volume},
    secrets=[modal.Secret.from_name("frameshift-secrets")],
    timeout=20 * 60,
    max_containers=1,
    scaledown_window=180,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=32, target_inputs=16)
class Server:
    @modal.enter(snap=True)
    def warm(self):
        """Load all CUDA models into a GPU memory snapshot once per deploy."""
        import torch

        from services import flow_service, sam2_service

        if not torch.cuda.is_available():
            raise RuntimeError("Modal attached no CUDA GPU to the FrameShift server")
        sam2_service.get_image_predictor()
        sam2_service.get_video_predictor()
        flow_service.get_raft_model(torch.device("cuda"))
        torch.cuda.synchronize()
        print(f"[Modal] warmed models on {torch.cuda.get_device_name(0)}")

    @modal.asgi_app()
    def fastapi_app(self):
        from main import app as fastapi

        return fastapi


@app.function(
    image=image,
    volumes={"/root/projects": volume},
    secrets=[modal.Secret.from_name("frameshift-secrets")],
    timeout=20 * 60,
)
def backfill_storage():
    """One-time migration from the Modal cache to Supabase Storage."""
    from services.supabase_storage_service import backfill_volume

    result = backfill_volume()
    volume.commit()
    return result
