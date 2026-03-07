import uuid
import json
from pathlib import Path

BASE_DIR = Path("/tmp/frameshift")

def create_project() -> dict:
    project_id = str(uuid.uuid4())[:8]
    project_dir = BASE_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "frames").mkdir(exist_ok=True)
    (project_dir / "masks").mkdir(exist_ok=True)
    (project_dir / "edited").mkdir(exist_ok=True)

    _write_status(project_id, {
        "status": "created",
        "frame_count": 0,
        "detections": {},
        "edit_status": "idle",
        "edit_progress": {"done": 0, "total": 0},
    })

    return {"project_id": project_id, "project_dir": str(project_dir)}

def get_project_dir(project_id: str) -> Path:
    project_dir = BASE_DIR / project_id
    if not project_dir.exists():
        raise FileNotFoundError(f"Project {project_id} not found")
    return project_dir

def _status_path(project_id: str) -> Path:
    return BASE_DIR / project_id / "status.json"

def _write_status(project_id: str, data: dict):
    path = _status_path(project_id)
    path.write_text(json.dumps(data))

def update_status(project_id: str, **kwargs):
    path = _status_path(project_id)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {}
    data.update(kwargs)
    path.write_text(json.dumps(data))

def get_status(project_id: str) -> dict:
    path = _status_path(project_id)
    if not path.exists():
        return {"status": "not_found"}
    return json.loads(path.read_text())
