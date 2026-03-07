import torch
from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

FRAME_PATH = BASE_DIR / "output" / "frame.jpg"

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolo11n.pt")

results = model(str(FRAME_PATH), device=device, conf=0.45, iou=0.4, imgsz=1280)

for result in results:
    boxes = result.boxes
    if boxes is None:
        continue

    print("\nDetected objects:")
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        label = model.names[cls_id]

        print({
            "label": label,
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(v, 2) for v in xyxy]
        })