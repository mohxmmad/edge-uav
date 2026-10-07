"""Download VisDrone-DET and build a small sampled subset for low-VRAM use.

Usage:
    python scripts/prepare_dataset.py --sample 800
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=800,
                   help="number of training images to keep (500-1000 recommended)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Use Ultralytics' bundled VisDrone config/URLs to locate-or-download
    try:
        from ultralytics.utils import SETTINGS
        datasets_dir = Path(SETTINGS.get("datasets_dir", DATA))
        print(f"Ultralytics datasets dir: {datasets_dir}")
    except Exception as e:
        print(f"[warn] {e}")
        datasets_dir = DATA

    # Locate VisDrone images (Ultralytics downloads to DATA/VisDrone or datasets dir)
    candidates = [p for p in list(DATA.glob("**/VisDrone*")) + list(
        datasets_dir.glob("**/VisDrone*")) if p.is_dir()]
    if not candidates:
        print("[info] VisDrone not found locally. Download it with:")
        print("       yolo task=detect mode=train model=yolov8n.pt data=VisDrone.yaml epochs=1 imgsz=640")
        print("       (it will auto-download). Then re-run this script.")
        return

    src = candidates[0]
    print(f"Found dataset at: {src}")

    train_imgs = sorted((src / "images" / "train").glob("*.jpg")) or \
                 sorted((src / "images" / "train").glob("*.*"))
    train_lbls = src / "labels" / "train"
    if not train_imgs:
        print(f"[error] No training images under {src}/images/train")
        return

    random.seed(args.seed)
    keep = set(random.sample(train_imgs, min(args.sample, len(train_imgs))))

    out = DATA / "VisDrone_sample"
    for split, imgs in [("train", sorted(keep)),
                        ("val", sorted((src / "images" / "val").glob("*.*"))[:200])]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = train_lbls / (img.stem + ".txt")
            if split == "val":
                lbl = src / "labels" / "val" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / lbl.name)

    # Class names from VisDrone
    names = ["pedestrian", "people", "bicycle", "car", "van", "truck",
             "tricycle", "awning-tricycle", "bus", "motor"]
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "nc": len(names), "names": names}
    (ROOT / "configs").mkdir(exist_ok=True)
    with open(ROOT / "configs" / "visdrone_sample.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Sample dataset ready: {out} ({len(keep)} train images)")
    print(f"Config written to configs/visdrone_sample.yaml")


if __name__ == "__main__":
    main()
