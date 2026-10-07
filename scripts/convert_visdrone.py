"""Convert downloaded VisDrone zips to Ultralytics YOLO layout."""
from pathlib import Path
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
vis = D / "VisDrone"

for src_name, split in [("VisDrone2019-DET-train", "train"),
                        ("VisDrone2019-DET-val", "val")]:
    src = D / src_name
    img_out = vis / "images" / split
    lbl_out = vis / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in (src / "images").glob("*.jpg"):
        shutil.move(str(img), img_out / img.name)

    anns = sorted((src / "annotations").glob("*.txt"))
    for i, f in enumerate(anns):
        img = img_out / f.with_suffix(".jpg").name
        w, h = Image.open(img).size
        dw, dh = 1.0 / w, 1.0 / h
        lines = []
        for row in [x.split(",") for x in f.read_text().strip().splitlines()]:
            if row[4] != "0":
                x, y, bw, bh = map(int, row[:4])
                cls = int(row[5]) - 1
                xc, yc = (x + bw / 2) * dw, (y + bh / 2) * dh
                wn, hn = bw * dw, bh * dh
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        (lbl_out / f.name).write_text("".join(lines))
        if i % 1000 == 0:
            print(split, i, "/", len(anns), flush=True)
    shutil.rmtree(src)
    print(split, "done", flush=True)

yaml = f"""path: {vis.resolve()}
train: images/train
val: images/val
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
(ROOT / "configs").mkdir(exist_ok=True)
(ROOT / "configs" / "visdrone.yaml").write_text(yaml)
print("wrote configs/visdrone.yaml")
