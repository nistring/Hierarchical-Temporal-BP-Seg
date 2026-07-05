"""Render prediction overlays for the 4-fold R1.3 runs (ssl + ultrasam, per vendor).

Per fold: load the model from lightning_logs/{tag}/, walk the held-out val COCO
grouped by video_id, run the model frame-by-frame with carried hidden state,
overlay via src.utils.process_video_stream, and write one mp4 per video.

Output: out/prediction_videos_4fold/{vendor}_fold{k}_ssl_ultrasam/video_{vid:03d}.mp4
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import yaml
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from src.model import TemporalSegmentationModel
from src.utils import load_model, post_processing, process_video_stream

IMG_ROOT = Path("data/SUIT/images")
LOG_ROOT = Path("lightning_logs")
OUT_ROOT = Path("out/prediction_videos_4fold")


def build_model(cfg_path: Path, ckpt_path: Path, device):
    cfg = yaml.safe_load(cfg_path.open())["model"].copy()
    cfg["image_size"] = tuple(cfg["image_size"])
    if "model_kwargs" in cfg:
        cfg.update(cfg.pop("model_kwargs"))
    if "kernel_size" in cfg:
        cfg["kernel_size"] = tuple(cfg["kernel_size"])
    model = load_model(TemporalSegmentationModel(**cfg), str(ckpt_path)).to(device).eval()
    return model, cfg["image_size"]


def render_one(model, image_size, frame_paths, out_path: Path, device, fps=10):
    H, W = image_size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    resize = v2.Resize(image_size)
    to_float = v2.ToDtype(torch.float32, scale=True)
    hidden = None
    for fp in frame_paths:
        gray = decode_image(str(fp), mode="GRAY").to(device)         # (1, h, w) uint8
        gray = resize(gray)                                          # (1, H, W) uint8
        x = to_float(gray)[None, None]                               # (1, 1, 1, H, W)
        with torch.no_grad():
            logits, hidden = model(x, hidden)
            prob = post_processing(logits[:, 0])[0]                  # (1+C, H, W)
        frame = process_video_stream(gray.float(), prob)             # (H, W, 3) torch
        writer.write(frame.cpu().numpy().astype("uint8"))
    writer.release()


def render_fold(vendor: str, fold: int, n_per_fold: int, device):
    tag = f"{vendor}_fold{fold}_ssl_ultrasam"
    ckpt = LOG_ROOT / tag / "checkpoints/last.ckpt"
    cfg = LOG_ROOT / tag / "config.yaml"
    if not ckpt.exists() or not cfg.exists():
        print(f"skip {tag}: missing ckpt/config")
        return
    coco = json.load(Path(f"data/SUIT/coco_annotations/{vendor}_fold{fold}_val_ultrasam.json").open())
    by_vid = defaultdict(list)
    for img in coco["images"]:
        by_vid[img["video_id"]].append(img)
    for v in by_vid.values():
        v.sort(key=lambda x: x["frame_id"])

    model, image_size = build_model(cfg, ckpt, device)
    out_dir = OUT_ROOT / tag
    for vid in tqdm(sorted(by_vid)[:n_per_fold], desc=tag):
        frame_paths = [IMG_ROOT / img["file_name"] for img in by_vid[vid]]
        render_one(model, image_size, frame_paths, out_dir / f"video_{vid:03d}.mp4", device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vendors", nargs="+", default=["ge", "mindray"])
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--n-per-fold", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    for v in args.vendors:
        for k in args.folds:
            render_fold(v, k, args.n_per_fold, device)


if __name__ == "__main__":
    main()
