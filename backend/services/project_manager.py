import uuid
import json
import threading
import tempfile
from pathlib import Path

BASE_DIR = Path(tempfile.gettempdir()) / "frameshift"

_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()

def _get_lock(project_id: str) -> threading.Lock:
    with _locks_lock:
        if project_id not in _locks:
            _locks[project_id] = threading.Lock()
        return _locks[project_id]


def create_project() -> dict:
    project_id = str(uuid.uuid4())[:8]
    project_dir = BASE_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "frames").mkdir(exist_ok=True)
    (project_dir / "masks").mkdir(exist_ok=True)
    (project_dir / "edited").mkdir(exist_ok=True)
    (project_dir / "backups").mkdir(exist_ok=True)

    _write_status(project_id, {
        "status": "created",
        "frame_count": 0,
        "detections": {},
        "edit_status": "idle",
        "edit_progress": {"done": 0, "total": 0},
        "ai_generation_id": None,
        "ai_preview_url": None,
        "ai_prompt": None,
        "ai_edit_status": None,
        "ai_edit_progress": None,
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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)

def update_status(project_id: str, **kwargs):
    lock = _get_lock(project_id)
    with lock:
        path = _status_path(project_id)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {}
        data.update(kwargs)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)

def get_status(project_id: str) -> dict:
    path = _status_path(project_id)
    if not path.exists():
        return {"status": "not_found"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"status": "loading"}
