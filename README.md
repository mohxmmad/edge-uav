# Edge AI-Based Object Detection for UAVs

Reproducible, lightweight computer-vision pipeline modeling an edge-deployment
workflow (train -> optimize -> benchmark) on a standard PC with zero budget.

## Structure
```
edge_uav/
├── configs/
│   └── visdrone_sample.yaml     # dataset config (sample subset)
├── data/                        # dataset goes here
├── models/                      # exported ONNX models
├── runs/                        # training outputs
├── scripts/
│   ├── prepare_dataset.py       # VisDrone-DET download + subsample
│   ├── train.py                 # fine-tune yolov8n
│   ├── export.py                # ONNX export (+FP16 fallback)
│   └── infer_benchmark.py       # ONNX Runtime benchmark + live overlay
├── requirements.txt
└── README.md
```

## Quick start
```bash
pip install -r requirements.txt

# 1. Download VisDrone and build a small sample (e.g. 800 images)
python scripts/prepare_dataset.py --sample 800

# 2. Train (CPU)
python scripts/train.py --epochs 30 --batch 8 --device cpu

# 3. Or low-VRAM GPU
python scripts/train.py --epochs 30 --batch 4 --device 0

# 4. Export to ONNX (simplified); add --fp16 for half precision
python scripts/export.py --weights runs/train/visdrone/weights/best.pt --fp16

# 5. Benchmark on a video or webcam
python scripts/infer_benchmark.py --source path/to/video.mp4 --model models/best.onnx
python scripts/infer_benchmark.py --source 0 --model models/best.onnx --imgsz 640
```

Note: `prepare_dataset.py` downloads VisDrone via Ultralytics' built-in
`VisDrone.yaml`. If the official URLs are unreachable, place the dataset
manually under `data/VisDrone` with the standard `images/` + `labels/` layout.
