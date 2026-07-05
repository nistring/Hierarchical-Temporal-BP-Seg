"""Generate pseudo-segmentation labels on a seed COCO using a teacher model.

Stateful per-video inference: hidden state propagates across frames within a
video, resets at video boundaries (matches training-time behavior with the
hierarchical ConvLSTM/ConvGRU stack). Frames where no foreground class has
sufficient confident pixel mass are dropped from the output COCO so the SSL
training distribution matches the trimmed labeled distribution.

Usage:
    PYTHONPATH=. python data/pseudo_label.py \\
        --domain ge \\
        --teacher-config lightning_logs/ge_fold0_finetune_ultrasam/config.yaml \\
        --teacher-ckpt lightning_logs/ge_fold0_finetune_ultrasam/checkpoints/last.ckpt \\
        --gpu 0 \\
        --conf-thresh 0.6 --min-pixels 64

To pseudo-label an arbitrary seed (e.g. orphans), override --seed/--out/--img-dir.

--conf-thresh accepts either one float (uniform) or eight comma-separated
floats (per-class for fg classes 1..8 = C5, C6, C7/MT, C8/LT, UT, SSN, AD, PD).

--entropy-thresh drops frames whose mean per-pixel softmax entropy across the
9 classes exceeds the threshold. Default `inf` = disabled. Useful range 1.0-1.8
(max possible is log(9)≈2.20). High entropy = teacher is uniformly uncertain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.model import TemporalSegmentationModel
from src.utils import load_model

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data/SUIT/coco_annotations"
IMG_ROOT = ROOT / "data/SUIT/images"

# Inverse of `category_match` in src/data_loader.py — pick the lowest CVAT id
# whose `category_match` value equals the model class. Model class -> CVAT cat.
#   data_loader category_match = {1:1, 2:2, 3:3, 4:4, 5:5, 6:3, 7:4, 8:6, 9:7, 10:8}
MODEL_CLASS_TO_CAT_ID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 9, 8: 10}


def parse_conf_thresh(spec: str, num_classes: int) -> list[float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) == 1:
        return parts * num_classes
    if len(parts) == num_classes:
        return parts
    raise ValueError(f"--conf-thresh must be 1 or {num_classes} comma-separated floats, got {len(parts)}")


def load_teacher(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg["model"]
    model = TemporalSegmentationModel(
        encoder_name=model_cfg["encoder_name"],
        segmentation_model_name=model_cfg["segmentation_model_name"],
        num_classes=model_cfg["num_classes"],
        temporal_model=model_cfg["temporal_model"],
        encoder_depth=model_cfg["encoder_depth"],
        temporal_depth=model_cfg["temporal_depth"],
        freeze_encoder=model_cfg.get("freeze_encoder", False),
        num_layers=model_cfg.get("num_layers", 1),
        kernel_size=tuple(model_cfg.get("kernel_size", [3, 3])),
        dilation=model_cfg.get("dilation", 1),
        conv_type=model_cfg.get("conv_type", "standard"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        use_hierarchical_fusion=model_cfg.get("use_hierarchical_fusion", True),
        **model_cfg.get("model_kwargs", {}),
    )
    model = load_model(model, str(ckpt_path))
    model.eval().to(device)
    return model, model_cfg


def mask_to_polygon(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if c.shape[0] >= 3:
            polys.append(c.flatten().tolist())
    return polys


def polygon_area_bbox(polys):
    pts = np.concatenate([np.array(p).reshape(-1, 2) for p in polys], axis=0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    area = float((x1 - x0) * (y1 - y0))
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], area


@torch.no_grad()
def pseudo_label(seed_path: Path, out_path: Path, img_dir: Path,
                 teacher_config: Path, teacher_ckpt: Path,
                 gpu: int, conf_thresh: list[float], min_pixels: int,
                 entropy_thresh: float, chunk: int):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    seed = json.loads(seed_path.read_text())
    model, model_cfg = load_teacher(teacher_config, teacher_ckpt, device)
    image_size = tuple(model_cfg["image_size"])
    num_classes = model_cfg["num_classes"]  # 8 fg
    if len(conf_thresh) != num_classes:
        raise ValueError(f"--conf-thresh count {len(conf_thresh)} != num_classes {num_classes}")

    # Group images by video, sorted by frame_id.
    by_vid: dict[int, list[dict]] = {}
    for im in seed["images"]:
        by_vid.setdefault(im["video_id"], []).append(im)
    for v in by_vid:
        by_vid[v].sort(key=lambda im: im["frame_id"])

    kept_images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    n_drop_lowconf = 0
    n_drop_entropy = 0
    n_total = 0
    tag = seed_path.stem

    for vid_id in tqdm(sorted(by_vid.keys()), desc=f"[{tag}] pseudo-labeling"):
        ims = by_vid[vid_id]
        frames = []
        for im in ims:
            arr = cv2.imread(str(img_dir / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            if arr.shape != image_size:
                arr = cv2.resize(arr, (image_size[1], image_size[0]),
                                 interpolation=cv2.INTER_AREA)
            frames.append(arr)
        t = torch.from_numpy(np.stack(frames)).unsqueeze(1).float() / 255.0

        hidden = None
        all_preds: list[torch.Tensor] = []
        for i in range(0, len(ims), chunk):
            x = t[i:i + chunk].unsqueeze(0).to(device)
            logits, hidden = model(x, hidden)
            probs = torch.softmax(logits, dim=2)
            all_preds.append(probs[0].cpu())
        probs = torch.cat(all_preds, dim=0).numpy()  # (T, 9, H, W)

        for im, p in zip(ims, probs):
            n_total += 1

            # Frame entropy gate (mean over pixels of -Σ p log p across the 9 classes).
            if math.isfinite(entropy_thresh):
                pp = np.clip(p, 1e-8, 1.0)
                ent = -(pp * np.log(pp)).sum(axis=0).mean()  # scalar
                if ent > entropy_thresh:
                    n_drop_entropy += 1
                    continue

            argmax = p.argmax(axis=0)  # (H, W) at image_size
            tgt_h, tgt_w = im["height"], im["width"]
            kept_any = False
            ann_buffer = []
            for k in range(1, num_classes + 1):
                cls_mask = (argmax == k)
                if cls_mask.sum() < min_pixels:
                    continue
                conf = p[k][cls_mask]
                if (conf > conf_thresh[k - 1]).sum() < min_pixels:
                    continue
                # Resize the binary class mask to COCO image dims so polygons
                # land in the right coord space when image dims != image_size.
                if (tgt_h, tgt_w) != image_size:
                    cls_mask_out = cv2.resize(
                        cls_mask.astype(np.uint8), (tgt_w, tgt_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    cls_mask_out = cls_mask.astype(np.uint8)
                polys = mask_to_polygon(cls_mask_out)
                if not polys:
                    continue
                bbox, area = polygon_area_bbox(polys)
                ann_buffer.append({
                    "id": 0,
                    "image_id": im["id"],
                    "video_id": vid_id,
                    "category_id": MODEL_CLASS_TO_CAT_ID[k],
                    "instance_id": 1,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": polys,
                    "iscrowd": 0,
                    "is_vid_train_frame": True,
                    "visibility": 1.0,
                    "occluded": False,
                    "truncated": False,
                })
                kept_any = True
            if not kept_any:
                n_drop_lowconf += 1
                continue
            kept_images.append(im)
            for a in ann_buffer:
                a["id"] = ann_id
                ann_id += 1
                annotations.append(a)

    kept_vid_ids = {im["video_id"] for im in kept_images}
    kept_videos = [v for v in seed["videos"] if v["id"] in kept_vid_ids]

    out_path.write_text(json.dumps({
        "categories": seed["categories"],
        "videos": kept_videos,
        "images": kept_images,
        "annotations": annotations,
    }))
    print(f"[{tag}] kept {len(kept_images)} / {n_total} frames "
          f"(low-conf dropped: {n_drop_lowconf}, entropy dropped: {n_drop_entropy}), "
          f"{len(annotations)} annotations across {len(kept_videos)} videos")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["ge", "mindray"],
                    help="Convenience: sets --seed, --out, --img-dir from this domain.")
    ap.add_argument("--seed", type=Path, help="Override seed COCO path.")
    ap.add_argument("--out", type=Path, help="Override output COCO path.")
    ap.add_argument("--img-dir", type=Path, help="Override image directory.")
    ap.add_argument("--teacher-config", type=Path, required=True)
    ap.add_argument("--teacher-ckpt", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--conf-thresh", type=str, default="0.6",
                    help="Single float for uniform threshold, or 8 comma-separated for per-class.")
    ap.add_argument("--min-pixels", type=int, default=64,
                    help="Minimum confident pixels per class to keep that class on a frame.")
    ap.add_argument("--entropy-thresh", type=float, default=float("inf"),
                    help="Drop frame if mean per-pixel entropy of softmax > this. Default disabled.")
    ap.add_argument("--chunk", type=int, default=50,
                    help="Frames per stateful chunk.")
    args = ap.parse_args()

    if args.domain:
        seed_path = args.seed or (COCO_DIR / f"{args.domain}_pseudo_seed.json")
        out_path = args.out or (COCO_DIR / f"{args.domain}_pseudo.json")
        img_dir = args.img_dir or (IMG_ROOT / f"{args.domain}_pseudo")
    else:
        if not (args.seed and args.out and args.img_dir):
            ap.error("If --domain is not given, --seed, --out, and --img-dir are all required.")
        seed_path, out_path, img_dir = args.seed, args.out, args.img_dir

    conf = parse_conf_thresh(args.conf_thresh, num_classes=8)
    pseudo_label(
        seed_path=seed_path, out_path=out_path, img_dir=img_dir,
        teacher_config=args.teacher_config, teacher_ckpt=args.teacher_ckpt,
        gpu=args.gpu, conf_thresh=conf, min_pixels=args.min_pixels,
        entropy_thresh=args.entropy_thresh, chunk=args.chunk,
    )


if __name__ == "__main__":
    main()
