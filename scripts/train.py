"""Fine-tune YOLOv8n on (sampled) VisDrone-DET.

Usage:
    python scripts/train.py --data configs/visdrone_sample.yaml --epochs 30 --batch 8 --device cpu
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "configs" / "visdrone_sample.yaml"))
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu", help="'cpu' or GPU id like 0")
    p.add_argument("--project", default=str(ROOT / "runs" / "train"))
    p.add_argument("--name", default="visdrone")
    args = p.parse_args()

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # low-VRAM friendly knobs
        workers=4,
        amp=True,
        cache=False,
    )

    # Standard object-detection metrics
    metrics = model.val()
    print("\n===== Validation metrics =====")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Best weights: {Path(args.project)/args.name/'weights'/'best.pt'}")


if __name__ == "__main__":
    main()
