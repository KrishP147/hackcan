import json
import subprocess
import shutil
from pathlib import Path

# Detect FFmpeg from system PATH
_FFMPEG = shutil.which("ffmpeg")
if _FFMPEG is None:
    # Fallback: try common installation paths
    import platform
    if platform.system() == "Windows":
        # Try common Windows locations
        possible_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        ]
        for path in possible_paths:
            if Path(path).exists():
                _FFMPEG = path
                break
    elif platform.system() == "Darwin":  # macOS
        # Try Homebrew location
        brew_path = Path("/opt/homebrew/bin/ffmpeg")
        if brew_path.exists():
            _FFMPEG = str(brew_path)
        else:
            brew_path = Path("/usr/local/bin/ffmpeg")
            if brew_path.exists():
                _FFMPEG = str(brew_path)
    
    if _FFMPEG is None:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: winget install FFmpeg\n"
            "  Linux: sudo apt-get install ffmpeg"
        )


_FFPROBE = str(Path(_FFMPEG).with_name("ffprobe")) if Path(str(Path(_FFMPEG).with_name("ffprobe"))).exists() else (shutil.which("ffprobe") or "ffprobe")


def probe_video(video_path: Path) -> dict:
    out = subprocess.run(
        [_FFPROBE, "-v", "quiet", "-print_format", "json",
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
    output_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)
    used_fps = fps if fps else info["fps"]
    vf = []
    if fps:  # only resample when explicitly forced
        vf.append(f"fps={fps}")
    if info["height"] > max_height:
        vf.append(f"scale=-2:{max_height}")
    cmd = [_FFMPEG, "-y", "-i", str(video_path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-qscale:v", "2", str(output_dir / "frame_%04d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(output_dir.glob("frame_*.jpg"))), used_fps


def encode_video(frames_dir: Path, output_path: Path, fps: float = 30) -> Path:
    cmd = [
        _FFMPEG, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
