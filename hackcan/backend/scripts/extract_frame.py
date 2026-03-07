import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

INPUT_VIDEO = BASE_DIR / "input" / "15454886_2560_1440_60fps.mp4"
OUTPUT_FRAME = BASE_DIR / "output" / "frame.jpg"

OUTPUT_FRAME.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    "ffmpeg",
    "-y",
    "-i", str(INPUT_VIDEO),
    "-vf", "select=eq(n\\,30)",
    "-vframes", "1",
    str(OUTPUT_FRAME),
]

subprocess.run(cmd, check=True)
print(f"Saved frame to {OUTPUT_FRAME}")