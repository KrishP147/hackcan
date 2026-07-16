import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

OUTPUT_FRAME = BASE_DIR / "output" / "frame.jpg"

parser = argparse.ArgumentParser(description="Extract a frame from a video")
parser.add_argument("input_video", type=Path)
args = parser.parse_args()

OUTPUT_FRAME.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    "ffmpeg",
    "-y",
    "-i", str(args.input_video),
    "-vf", "select=eq(n\\,30)",
    "-vframes", "1",
    str(OUTPUT_FRAME),
]

subprocess.run(cmd, check=True)
print(f"Saved frame to {OUTPUT_FRAME}")
