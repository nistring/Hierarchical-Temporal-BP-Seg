"""Score the saved SAM2 video-predictor COCO against manual GT and UltraSam GT.

Reads `data/SUIT/coco_annotations/sonosite_val_sam2_video.json` (produced by
`sam2_video_predict.py`) and computes:

  * `manual_mean_iou` + `manual_per_category` — IoU vs manual polygons on the
    319 manually-annotated frames. Empty SAM2 predictions on those frames are
    scored as IoU=0 (not skipped).
  * `mean_iou_score` + `iou_per_category` — dense per-(video, category) IoU
    vs UltraSam labels, aggregated by pooling TP/FP/FN over every val frame
    in the (video, category) pair. Captures propagation hallucinations that
    the sparse manual metric cannot see.
  * `fp_rate_on_absent_frames` — fraction of (video, category, frame) triples
    where UltraSam has no GT for the category but SAM2 emits a non-empty
    mask. Direct measure of SAM2's "keep tracking after the structure
    disappears" failure mode.
  * `temporal_consistency_mean` — mean per-(video, category) binary-mask L1
    between consecutive frames, averaged across videos. Comparable to the
    trainer's `temporal_consistency_mean`.

CPU-only; runs in seconds. Output: `out/sam2_video_eval_metrics.json`.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as cocomask
from pycocotools.coco import COCO

REPO = Path(__file__).resolve().parent.parent
PRED_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val_sam2_video.json"
MANUAL_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val_manual.json"
DENSE_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val_ultrasam.json"
OUT = REPO / "out" / "sam2_video_eval_metrics.json"


def poly_to_mask(seg, h, w):
    if not seg:
        return None
    if isinstance(seg, list):
        rles = cocomask.frPyObjects(seg, h, w)
        rle = cocomask.merge(rles) if isinstance(rles, list) else rles
    else:
        rle = seg
    return cocomask.decode(rle).astype(np.uint8)


def index_by_image_cat(coco: COCO):
    """(image_id, cat_id) -> list of annotation dicts."""
    out = defaultdict(list)
    for ann in coco.anns.values():
        out[(ann["image_id"], ann["category_id"])].append(ann)
    return out


def union_mask(anns, h, w):
    """Union of all polygons in `anns` -> binary mask. Empty if no segs."""
    m = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        sub = poly_to_mask(ann.get("segmentation"), h, w)
        if sub is not None:
            m |= sub
    return m


def main():
    pred = COCO(str(PRED_COCO))
    manual = COCO(str(MANUAL_COCO))
    dense = COCO(str(DENSE_COCO))

    pred_idx = index_by_image_cat(pred)
    manual_idx = index_by_image_cat(manual)
    dense_idx = index_by_image_cat(dense)

    # Map: video_id -> list of (frame_id, image_id, h, w), sorted by frame_id
    per_vid = defaultdict(list)
    for im in pred.imgs.values():
        per_vid[im["video_id"]].append((im["frame_id"], im["id"],
                                        im["height"], im["width"]))
    for v in per_vid:
        per_vid[v].sort()

    # All COCO categories
    cat_ids = sorted(pred.cats.keys())

    # ---- manual mIoU ----
    manual_scores_by_cat = defaultdict(list)
    for (img_id, cat_id), anns in manual_idx.items():
        gt_anns = [a for a in anns if a.get("segmentation")]
        if not gt_anns:
            continue
        im = manual.imgs[img_id]
        gt = union_mask(gt_anns, im["height"], im["width"])
        pr_anns = pred_idx.get((img_id, cat_id), [])
        pr = union_mask(pr_anns, im["height"], im["width"])
        inter = int(np.logical_and(pr, gt).sum())
        union = int(np.logical_or(pr, gt).sum())
        iou = inter / union if union > 0 else 0.0
        manual_scores_by_cat[cat_id].append(iou)

    manual_all = [s for v in manual_scores_by_cat.values() for s in v]
    manual_mean = float(np.mean(manual_all)) if manual_all else 0.0

    # ---- dense mIoU + FP-on-absent ----
    pair_tp = defaultdict(int)
    pair_fp = defaultdict(int)
    pair_fn = defaultdict(int)
    absent_total = 0
    absent_with_pred = 0
    pair_has_pred_or_gt = set()  # pairs we should report on

    for video_id, frames in per_vid.items():
        for cat_id in cat_ids:
            for fid, img_id, H, W in frames:
                pr_anns = pred_idx.get((img_id, cat_id), [])
                gt_anns = [a for a in dense_idx.get((img_id, cat_id), [])
                           if a.get("segmentation")]
                pr = union_mask(pr_anns, H, W) if pr_anns else np.zeros((H, W), dtype=np.uint8)
                if not gt_anns:
                    # GT absent for this (video, cat, frame). Any pred pixel is FP.
                    fp = int(pr.sum())
                    if pr_anns or fp > 0:
                        pair_fp[(video_id, cat_id)] += fp
                        pair_has_pred_or_gt.add((video_id, cat_id))
                    absent_total += 1
                    if fp > 0:
                        absent_with_pred += 1
                else:
                    gt = union_mask(gt_anns, H, W)
                    tp = int(np.logical_and(pr, gt).sum())
                    fp = int(np.logical_and(pr, np.logical_not(gt)).sum())
                    fn = int(np.logical_and(np.logical_not(pr), gt).sum())
                    pair_tp[(video_id, cat_id)] += tp
                    pair_fp[(video_id, cat_id)] += fp
                    pair_fn[(video_id, cat_id)] += fn
                    pair_has_pred_or_gt.add((video_id, cat_id))

    pair_ious = []
    cat_ious = defaultdict(list)
    for (v, c) in pair_has_pred_or_gt:
        tp = pair_tp.get((v, c), 0)
        fp = pair_fp.get((v, c), 0)
        fn = pair_fn.get((v, c), 0)
        denom = tp + fp + fn
        if denom == 0:
            continue
        iou = tp / denom
        pair_ious.append(iou)
        cat_ious[c].append(iou)

    dense_mean = float(np.mean(pair_ious)) if pair_ious else 0.0
    fp_rate = absent_with_pred / absent_total if absent_total else 0.0

    # ---- temporal drift ----
    drift_per_video = []
    for video_id, frames in per_vid.items():
        per_cat_drift = []
        for cat_id in cat_ids:
            prev = None
            sums = 0.0
            n = 0
            for fid, img_id, H, W in frames:
                pr_anns = pred_idx.get((img_id, cat_id), [])
                pr = union_mask(pr_anns, H, W) if pr_anns else np.zeros((H, W), dtype=np.uint8)
                if prev is not None:
                    sums += float(np.abs(pr.astype(np.int16) - prev.astype(np.int16)).mean())
                    n += 1
                prev = pr
            if n > 0:
                per_cat_drift.append(sums / n)
        if per_cat_drift:
            drift_per_video.append(float(np.mean(per_cat_drift)))
    drift_mean = float(np.mean(drift_per_video)) if drift_per_video else 0.0

    cat_names = {1: "C5", 2: "C6", 3: "C7", 4: "C8", 5: "UT",
                 6: "MT", 7: "LT", 8: "SSN", 9: "AD", 10: "PD"}

    out = {
        "manual_mean_iou": manual_mean,
        "manual_per_category": {
            cat_names.get(c, str(c)): {
                "n": len(manual_scores_by_cat[c]),
                "mean_iou": float(np.mean(manual_scores_by_cat[c])),
            } for c in sorted(manual_scores_by_cat)
        },
        "mean_iou_score": dense_mean,
        "iou_per_category": {cat_names.get(c, str(c)): float(np.mean(cat_ious[c]))
                             for c in sorted(cat_ious)},
        "fp_rate_on_absent_frames": fp_rate,
        "n_absent_frame_cat_pairs": absent_total,
        "n_absent_with_hallucination": absent_with_pred,
        "temporal_consistency_mean": drift_mean,
        "n_pairs_dense": len(pair_ious),
        "source_pred_coco": str(PRED_COCO),
        "source_dense_gt": str(DENSE_COCO),
        "source_manual_gt": str(MANUAL_COCO),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
