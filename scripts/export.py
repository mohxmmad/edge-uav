"""Export trained YOLOv8 weights to ONNX with graph simplification (+optional FP16).

Usage:
    python scripts/export.py --weights runs/train/visdrone/weights/best.pt
    python scripts/export.py --weights ... --fp16
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fp16", action="store_true",
                   help="export in FP16 to halve size / memory (GPU-friendly)")
    p.add_argument("--out", default=str(ROOT / "models"))
    args = p.parse_args()

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        half=args.fp16,
        dynamic=False,
    )
    onnx_path = Path(onnx_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ("best_fp16.onnx" if args.fp16 else "best.onnx")
    shutil.move(str(onnx_path), dest)

    size_mb = dest.stat().st_size / 1e6
    print(f"Exported: {dest} ({size_mb:.2f} MB)")
    try:
        import onnx
        m = onnx.load(str(dest))
        onnx.checker.check_model(m)
        print("ONNX model check: OK")
    except Exception as e:
        print(f"[warn] ONNX check skipped: {e}")


if __name__ == "__main__":
    main()
