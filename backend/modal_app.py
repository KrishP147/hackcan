"""Modal deploy (Doc 1 §9). No custom native builds on this path —
Backend A is torch-native; Doc 2's CUDA kernels are dev-only and flag-guarded.

Deploy:  modal deploy modal_app.py   (requires `modal token new` first)
Secrets: create `frameshift-secrets` with GEMINI_API_KEY + CLOUDINARY_URL.
"""
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
    secrets=[modal.Secret.from_name("frameshift-secrets")],
    scaledown_window=120,
    enable_memory_snapshot=True,
)
class Server:
    @modal.enter(snap=True)
    def warm(self):
        """Load SAM 2 + RAFT once; memory snapshots restore a warmed process."""
        import torch
        from services import flow_service, sam2_service
        sam2_service.get_image_predictor()
        flow_service._build_raft(torch.device("cuda"))

    @modal.asgi_app()
    def fastapi_app(self):
        import os
        os.environ.setdefault("RATE_LIMIT", "1")
        from main import app as fastapi
        return fastapi
