"""Compute FP-on-absent rate (and dense IoU, temporal drift) for our model,
using the SAME protocol as `data/sam2_video_metrics.py` so the numbers in
rebuttal R1.1 are apples-to-apples with SAM2.

Loads `seq50_ultrasam` from checkpoint, walks every val frame with hidden
state continuous per video, applies the same argmax-then-threshold pipeline
the trainer's `test_step` uses (`src/model.py:402-407`), and for each
(video, COCO category c, frame) checks:

  * UltraSam GT empty for c at this frame?  (absence)
  * Our model emits non-empty mask for class `category_match[c]`?
    (`category_match` from `src/data_loader.py:13` merges the 10 COCO
    categories into 8 model classes.)

If both: hallucinated FP. Report:
  - fp_rate_on_absent_frames
  - dense mIoU per (video, model class) pooled across frames
  - temporal drift across the softmax probability tensor (same as the
    trainer's `temporal_consistency_mean`).

Run:
    /home/nistring/.conda/envs/suit/bin/python data/model_dense_metrics.py --gpu 0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from torchvision.transforms import v2
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data_loader import category_match  # noqa: E402
from src.model import SegmentationTrainer  # noqa: E402
from src.utils import post_processing  # noqa: E402

CKPT = REPO / "lightning_logs" / "seq50_ultrasam" / "checkpoints" / "last.ckpt"
CFG = REPO / "lightning_logs" / "seq50_ultrasam" / "config.yaml"
IMG_ROOT = REPO / "data" / "SUIT" / "images" / "sonosite_val"
SEED_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val.json"
DENSE_GT = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val_ultrasam.json"
OUT = REPO / "out" / "ours_dense_metrics.json"


def poly_to_mask(seg, h, w):
    if not seg:
        return None
    if isinstance(seg, list):
        rles = cocomask.frPyObjects(seg, h, w)
        rle = cocomask.merge(rles) if isinstance(rles, list) else rles
    else:
        rle = seg
    return cocomask.decode(rle).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    cfg = yaml.safe_load(open(CFG))
    img_size = tuple(cfg["model"]["image_size"])

    # Build the model the way main.py does, then load the lightning checkpoint.
    from main import main as _main  # noqa
    # Easier path: re-use SegmentationTrainer.load_from_checkpoint via main.py
    # would launch full Trainer; instead, build minimal model and load weights.
    from src.model import TemporalSegmentationModel
    mcfg = cfg["model"]
    model = TemporalSegmentationModel(
        encoder_name=mcfg["encoder_name"],
        segmentation_model_name=mcfg["segmentation_model_name"],
        num_classes=mcfg["num_classes"],
        temporal_model=mcfg["temporal_model"],
        encoder_depth=mcfg["encoder_depth"],
        temporal_depth=mcfg["temporal_depth"],
        freeze_encoder=mcfg.get("freeze_encoder", False),
        num_layers=mcfg.get("num_layers", 1),
        kernel_size=tuple(mcfg.get("kernel_size", [3, 3])),
        dilation=mcfg.get("dilation", 1),
        conv_type=mcfg.get("conv_type", "standard"),
        encoder_weights=None,  # don't redownload — we'll load ckpt weights
        use_hierarchical_fusion=mcfg.get("use_hierarchical_fusion", True),
        **mcfg.get("model_kwargs", {}),
    )
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Lightning prepends "model."
    model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print(f"WARN missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"WARN unexpected keys (first 3): {unexpected[:3]}")
    model = model.eval().to(device)

    # Preprocess: same as data_loader / SegmentationTrainer expects
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.Resize(img_size),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Load val COCO + GT
    seed_coco = COCO(str(SEED_COCO))
    dense_coco = COCO(str(DENSE_GT))

    # Group val frames per video, sorted by frame_id
    per_vid = defaultdict(list)
    for im in seed_coco.imgs.values():
        per_vid[im["video_id"]].append(
            (im["frame_id"], im["id"], im["height"], im["width"], im["file_name"])
        )
    for v in per_vid:
        per_vid[v].sort()

    # Index dense GT: (image_id, cat_id) -> annotation list
    dense_idx = defaultdict(list)
    for ann in dense_coco.anns.values():
        if ann.get("segmentation"):
            dense_idx[(ann["image_id"], ann["category_id"])].append(ann)

    # Accumulators
    n_model_classes = mcfg["num_classes"]  # 8
    cat_ids = sorted(dense_coco.cats.keys())  # 1..10

    pair_tp = defaultdict(int)  # (video, model_class) -> tp pixels
    pair_fp = defaultdict(int)
    pair_fn = defaultdict(int)

    absent_total = 0
    absent_with_pred = 0
    # FP-on-absent at COCO category resolution (matches SAM2 protocol)
    cat_absent_total = defaultdict(int)
    cat_absent_pred = defaultdict(int)

    drift_per_video = []

    with torch.no_grad():
        for video_id, frames in tqdm(sorted(per_vid.items()), desc="videos"):
            hidden = None
            prev_probs = None
            sum_drift = 0.0
            n_drift = 0

            for fid, image_id, H, W, fname in frames:
                img = cv2.imread(str(IMG_ROOT / fname), cv2.IMREAD_GRAYSCALE)
                # Normalize and feed
                x = preprocess(torch.from_numpy(img)[None])  # (1, h, w)
                x = x.unsqueeze(0).unsqueeze(0).to(device)  # (B=1, T=1, C=1, H, W)
                logits, hidden = model(x, hidden)
                # logits: (1, 1, 9, h, w). post_processing softmax+something
                probs = post_processing(logits[0])  # (T=1, 9, h, w) — interp & softmax
                probs_fg = probs[:, 1:]  # drop BG -> (1, 8, h, w)
                # Resize predictions to original frame resolution (H, W)
                probs_fg = F.interpolate(probs_fg, size=(H, W), mode="bilinear",
                                         align_corners=True)
                # Same argmax+threshold as test_step
                argmax = torch.argmax(probs_fg, dim=1, keepdim=True)
                summed = torch.sum(probs_fg, dim=1, keepdim=True)
                preds = torch.zeros_like(probs_fg)
                preds.scatter_(1, argmax, summed)
                preds_bin = (preds > 0.5).cpu().numpy()[0]  # (8, H, W) bool

                # Temporal drift: full softmax probability tensor frame-to-frame L1
                # (same definition as trainer: probs.shape = (B,T,C,H,W))
                probs_np = probs.cpu().numpy()[0]  # (9, h, w)
                if prev_probs is not None:
                    sum_drift += float(np.abs(probs_np - prev_probs).mean())
                    n_drift += 1
                prev_probs = probs_np

                # Per-(video, model_class) TP/FP/FN against UltraSam GT
                # GT per model class = union of UltraSam masks for COCO cats
                # that map to that model class.
                gt_by_mc = {mc: np.zeros((H, W), dtype=np.uint8)
                            for mc in range(1, n_model_classes + 1)}
                for c in cat_ids:
                    mc = category_match[c]
                    for ann in dense_idx.get((image_id, c), []):
                        sub = poly_to_mask(ann["segmentation"], H, W)
                        if sub is not None:
                            gt_by_mc[mc] |= sub

                for mc in range(1, n_model_classes + 1):
                    pr = preds_bin[mc - 1].astype(np.uint8)
                    gt = gt_by_mc[mc]
                    if gt.sum() == 0:
                        # absent for this model class at this frame
                        pair_fp[(video_id, mc)] += int(pr.sum())
                    else:
                        tp = int(np.logical_and(pr, gt).sum())
                        fp = int(np.logical_and(pr, np.logical_not(gt)).sum())
                        fn = int(np.logical_and(np.logical_not(pr), gt).sum())
                        pair_tp[(video_id, mc)] += tp
                        pair_fp[(video_id, mc)] += fp
                        pair_fn[(video_id, mc)] += fn

                # FP-on-absent at the COCO category resolution (matches the
                # SAM2 metric in `sam2_video_metrics.py`). For each c in 1..10:
                # the (video, c, frame) triple is "absent" if dense GT for c
                # at this frame is empty; we say "predicted" if our model
                # emitted any pixel for the corresponding model class.
                for c in cat_ids:
                    has_gt = bool(dense_idx.get((image_id, c)))
                    if has_gt:
                        continue
                    absent_total += 1
                    cat_absent_total[c] += 1
                    mc = category_match[c]
                    if preds_bin[mc - 1].any():
                        absent_with_pred += 1
                        cat_absent_pred[c] += 1

            if n_drift > 0:
                drift_per_video.append(sum_drift / n_drift)

    # Aggregate dense mIoU per (video, model_class)
    pair_ious = []
    mc_ious = defaultdict(list)
    pair_pairs = set(list(pair_tp.keys()) + list(pair_fp.keys()) + list(pair_fn.keys()))
    for (v, mc) in pair_pairs:
        tp = pair_tp.get((v, mc), 0)
        fp = pair_fp.get((v, mc), 0)
        fn = pair_fn.get((v, mc), 0)
        denom = tp + fp + fn
        if denom == 0:
            continue
        iou = tp / denom
        pair_ious.append(iou)
        mc_ious[mc].append(iou)

    cat_names = {1: "C5", 2: "C6", 3: "C7", 4: "C8", 5: "UT",
                 6: "MT", 7: "LT", 8: "SSN", 9: "AD", 10: "PD"}
    model_class_names = {1: "C5", 2: "C6", 3: "C7/MT", 4: "C8/LT",
                         5: "UT", 6: "SSN", 7: "AD", 8: "PD"}

    out = {
        "mean_iou_score": float(np.mean(pair_ious)) if pair_ious else 0.0,
        "iou_per_model_class": {model_class_names[mc]: float(np.mean(mc_ious[mc]))
                                for mc in sorted(mc_ious)},
        "fp_rate_on_absent_frames": absent_with_pred / absent_total if absent_total else 0.0,
        "n_absent_frame_cat_pairs": int(absent_total),
        "n_absent_with_hallucination": int(absent_with_pred),
        "fp_rate_per_cat": {cat_names[c]: (cat_absent_pred[c] / cat_absent_total[c]
                                            if cat_absent_total[c] else 0.0)
                            for c in cat_ids},
        "temporal_consistency_mean": float(np.mean(drift_per_video)) if drift_per_video else 0.0,
        "n_pairs_dense": len(pair_ious),
        "notes": (
            "Dense IoU computed per (video, model_class) by pooling TP/FP/FN "
            "across every val frame; GT per model class is the union of the "
            "UltraSam masks for the COCO categories that map to that model "
            "class via src/data_loader.py:category_match (C7/MT merged into "
            "model class 3, C8/LT merged into model class 4). FP-on-absent "
            "rate is computed at the COCO-category resolution (10 categories) "
            "to match `data/sam2_video_metrics.py` exactly: a (video, COCO "
            "cat c, frame) triple is 'absent' when UltraSam has no annotation "
            "for c at that frame, 'predicted' when our model emits any pixel "
            "for category_match[c]."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
