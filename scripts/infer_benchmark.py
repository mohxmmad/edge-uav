"""ONNX Runtime inference + benchmark on video/webcam with live overlay.

Usage:
    python scripts/infer_benchmark.py --source video.mp4 --model models/best.onnx
    python scripts/infer_benchmark.py --source 0 --model models/best_fp16.onnx
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (nw, nh))
    top, bottom = (size - nh) // 2, (size - nh) - (size - nh) // 2
    left, right = (size - nw) // 2, (size - nw) - (size - nw) // 2
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (left, top)


def postprocess(output, r, pad, conf, iou, orig_shape):
    preds = np.squeeze(output[0]).T  # (N, 4+nc)
    boxes, scores, cls = [], [], []
    for p in preds:
        obj = p[4:]
        c = int(np.argmax(obj)); s = float(obj[c])
        if s < conf:
            continue
        cx, cy, w, h = p[:4]
        x1 = (cx - w / 2 - pad[0]) / r
        y1 = (cy - h / 2 - pad[1]) / r
        x2 = (cx + w / 2 - pad[0]) / r
        y2 = (cy + h / 2 - pad[1]) / r
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(s); cls.append(c)
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    return [(boxes[i], scores[i], cls[i]) for i in (idx.flatten() if len(idx) else [])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="video path or webcam index")
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    sess = ort.InferenceSession(args.model,
                                providers=["CUDAExecutionProvider",
                                           "CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    in_dtype = np.float16 if "float16" in inp.type else np.float32

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    latencies = []
    frames = 0
    t_start = time.perf_counter()
    print(f"Model: {args.model} | dtype: {in_dtype} | input: {inp.shape}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        blob, r, pad = letterbox(frame, args.imgsz)
        blob = blob.astype(in_dtype) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        t0 = time.perf_counter()
        out = sess.run([out_name], {inp.name: blob})
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)
        fps = 1000.0 / lat_ms

        for box, score, c in postprocess(out, r, pad, args.conf, args.iou, frame.shape):
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{NAMES[c]} {score:.2f}", (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{lat_ms:.1f} ms | {fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("edge_uav", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start

    # Parameter count from ONNX graph initializers
    import onnx
    m = onnx.load(args.model)
    n_params = sum(int(np.prod([d.dim_value for d in i.dims])) for i in m.graph.initializer)
    mean_lat = float(np.mean(latencies)) if latencies else 0
    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Parameters : {n_params/1e6:.2f} M")
    print(f"Model size : {Path(args.model).stat().st_size/1e6:.2f} MB")
    print(f"Mean latency: {mean_lat:.2f} ms")
    print(f"Avg FPS    : {frames/elapsed:.2f} ({frames} frames in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
