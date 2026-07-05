"""SAM2 video predictor on sonosite_val — predict and save (R1.1).

Runs `sam2.1_l.pt` via Ultralytics' `SAM2VideoPredictor` in *video propagator*
mode (one bbox prompt per (video, category) at the earliest CVAT-tracked frame,
forward+backward memory-bank propagation) and saves the propagated masks as
a COCO-format JSON at `data/SUIT/coco_annotations/sonosite_val_sam2_video.json`.

Output schema matches the autolabel COCOs (`*_sam2.json`, `*_ultrasam.json`):
copies the source `sonosite_val.json` images + categories, then replaces the
annotation list with SAM2 video-predictor predictions. Each prediction
annotation has `image_id`, `category_id`, polygon `segmentation` (largest
contour of the propagated mask via `cv2.findContours`), `bbox`, and `area`.
Frames where SAM2 emits no mask, or where the contour has <3 vertices, get
`segmentation: []` so absent/failed predictions remain explicit in the file.

Score the saved file with `data/sam2_video_metrics.py` (CPU-only, seconds).

Run:
    /home/nistring/.conda/envs/suit/bin/python data/sam2_video_predict.py --gpu 0
"""

import argparse
import copy
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from ultralytics.models.sam import SAM2VideoPredictor

REPO = Path(__file__).resolve().parent.parent
IMG_ROOT = REPO / "data" / "SUIT" / "images" / "sonosite_val"
SEED_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val.json"
MANUAL_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val_manual.json"
OUT_COCO = REPO / "data" / "SUIT" / "coco_annotations" / "sonosite_val_sam2_video.json"


def collect_video_frames(seed_coco: COCO):
    per_vid = defaultdict(list)
    for im in seed_coco.imgs.values():
        per_vid[im["video_id"]].append((im["frame_id"], IMG_ROOT / im["file_name"], im))
    for v in per_vid:
        per_vid[v].sort(key=lambda x: x[0])
    return per_vid


def seed_bbox_for(seed_coco: COCO, video_id: int, cat_id: int):
    candidates = []
    for ann in seed_coco.anns.values():
        if ann["category_id"] != cat_id:
            continue
        im = seed_coco.imgs[ann["image_id"]]
        if im["video_id"] != video_id:
            continue
        x, y, w, h = ann["bbox"]
        candidates.append((im["frame_id"], [x, y, x + w, y + h]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0]


def manual_bbox_fallback(manual_coco: COCO, video_id: int, cat_id: int):
    """For (video, cat) pairs the upstream tracker missed entirely, derive the
    seed bbox from the earliest manual-mask annotation. Same xyxy format."""
    from pycocotools import mask as cm
    candidates = []
    for ann in manual_coco.anns.values():
        if ann["category_id"] != cat_id or not ann.get("segmentation"):
            continue
        im = manual_coco.imgs[ann["image_id"]]
        if im["video_id"] != video_id:
            continue
        rles = cm.frPyObjects(ann["segmentation"], im["height"], im["width"])
        rle = cm.merge(rles) if isinstance(rles, list) else rles
        msk = cm.decode(rle).astype(np.uint8)
        ys, xs = np.where(msk > 0)
        if len(xs) == 0:
            continue
        candidates.append((im["frame_id"],
                           [int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max())]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0]


def _write_temp_mp4(frame_paths, out_path):
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))
    for p in frame_paths:
        writer.write(cv2.imread(str(p)))
    writer.release()


def propagate(predictor, frame_paths, seed_bbox, tmp_dir):
    mp4 = Path(tmp_dir) / "clip.mp4"
    _write_temp_mp4(frame_paths, mp4)
    results = predictor(source=str(mp4), bboxes=[seed_bbox], labels=[1])
    masks = []
    for r in results:
        if r.masks is None or len(r.masks) == 0:
            masks.append(None)
        else:
            masks.append(r.masks.data[0].cpu().numpy().astype(np.uint8))
    mp4.unlink(missing_ok=True)
    return masks


def _resize(m, h, w):
    if m is None:
        return None
    if m.shape == (h, w):
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def mask_to_coco_annotation(mask, image_id, category_id, ann_id):
    """Convert a binary mask to a COCO annotation. Returns the annotation dict;
    `segmentation` is `[]` if the mask is empty or its largest contour has <3
    vertices. Polygon comes from `cv2.findContours(RETR_EXTERNAL,
    CHAIN_APPROX_SIMPLE)` — same convention as `autolabel_ultrasam.py`."""
    ann = {
        "id": ann_id,
        "image_id": int(image_id),
        "category_id": int(category_id),
        "iscrowd": 0,
        "segmentation": [],
        "bbox": [0, 0, 0, 0],
        "area": 0,
    }
    if mask is None or mask.sum() == 0:
        return ann
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ann
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return ann
    poly = contour.flatten().tolist()
    ys, xs = np.where(mask > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    ann["segmentation"] = [poly]
    ann["bbox"] = [x1, y1, int(x2 - x1), int(y2 - y1)]
    ann["area"] = int(mask.sum())
    return ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-videos", type=int, default=None)
    args = parser.parse_args()

    OUT_COCO.parent.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"

    seed_coco = COCO(str(SEED_COCO))
    manual_coco = COCO(str(MANUAL_COCO))
    per_vid_frames = collect_video_frames(seed_coco)

    # All (video, cat) pairs to propagate: any category that appears in EITHER
    # the seed COCO or the manual COCO for that video.
    pairs = set()
    for ann in seed_coco.anns.values():
        im = seed_coco.imgs[ann["image_id"]]
        pairs.add((im["video_id"], ann["category_id"]))
    for ann in manual_coco.anns.values():
        if ann.get("segmentation"):
            im = manual_coco.imgs[ann["image_id"]]
            pairs.add((im["video_id"], ann["category_id"]))
    pairs = sorted(pairs)
    if args.limit_videos:
        keep = sorted({v for v, _ in pairs})[: args.limit_videos]
        pairs = [(v, c) for (v, c) in pairs if v in keep]

    predictor = SAM2VideoPredictor(overrides={
        "task": "segment", "mode": "predict", "model": "sam2.1_l.pt",
        "imgsz": 1024, "save": False, "verbose": False, "device": device,
    })

    # Map (video, frame_id) -> image meta for output annotations.
    frame_to_image_id = {}
    for im in seed_coco.imgs.values():
        frame_to_image_id[(im["video_id"], im["frame_id"])] = im

    new_anns = []
    next_ann_id = 1
    tmp_root = tempfile.mkdtemp(prefix="sam2_video_predict_")

    for video_id, cat_id in tqdm(pairs, desc="video,cat pairs"):
        frames = per_vid_frames[video_id]
        if not frames:
            continue
        frame_ids = [f[0] for f in frames]
        paths = [f[1] for f in frames]
        im_meta = frames[0][2]
        H, W = im_meta["height"], im_meta["width"]

        seed = seed_bbox_for(seed_coco, video_id, cat_id)
        if seed is None:
            seed = manual_bbox_fallback(manual_coco, video_id, cat_id)
            if seed is None:
                continue
        seed_frame, seed_box = seed
        if seed_frame not in frame_ids:
            continue
        i_seed = frame_ids.index(seed_frame)

        predictor.inference_state = {}
        fwd_paths = paths[i_seed:]
        fwd_fids = frame_ids[i_seed:]
        fwd_masks = propagate(predictor, fwd_paths, seed_box, tmp_root)

        predictor.inference_state = {}
        if i_seed > 0:
            bwd_paths = list(reversed(paths[: i_seed + 1]))
            bwd_fids = list(reversed(frame_ids[: i_seed + 1]))
            bwd_masks = propagate(predictor, bwd_paths, seed_box, tmp_root)
        else:
            bwd_masks, bwd_fids = [], []

        pred_by_frame = {}
        for fid, m in zip(bwd_fids, bwd_masks):
            pred_by_frame[fid] = m
        for fid, m in zip(fwd_fids, fwd_masks):
            pred_by_frame[fid] = m

        # Write one annotation per (frame_id) for this (video, cat). We keep
        # empty/no-mask entries explicitly with `segmentation: []` so consumers
        # can distinguish "SAM2 returned nothing" from "frame was not visited".
        for fid in frame_ids:
            pred = pred_by_frame.get(fid)
            pred_rs = _resize(pred, H, W)
            im = frame_to_image_id.get((video_id, fid))
            if im is None:
                continue
            ann = mask_to_coco_annotation(pred_rs, im["id"], cat_id, next_ann_id)
            new_anns.append(ann)
            next_ann_id += 1

    # Assemble output COCO: reuse images + categories from seed COCO.
    out_coco = {
        "info": seed_coco.dataset.get("info", {}),
        "licenses": seed_coco.dataset.get("licenses", []),
        "images": copy.deepcopy(seed_coco.dataset["images"]),
        "categories": copy.deepcopy(seed_coco.dataset["categories"]),
        "annotations": new_anns,
        "_protocol": (
            "SAM2 video predictor (Ultralytics SAM2VideoPredictor, sam2.1_l.pt). "
            "Per (video, category): seed bbox from earliest seed-COCO frame, "
            "forward+backward propagation over full video. One annotation per "
            "(image, category) pair the propagation visited; empty/failed "
            "predictions kept with segmentation:[]. Polygons are the largest "
            "contour from cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)."
        ),
    }
    with open(OUT_COCO, "w") as f:
        json.dump(out_coco, f)
    n_nonempty = sum(1 for a in new_anns if a["segmentation"])
    print(f"Wrote {OUT_COCO}")
    print(f"  total annotations: {len(new_anns):,}")
    print(f"  non-empty (mask present): {n_nonempty:,} "
          f"({100 * n_nonempty / max(len(new_anns), 1):.1f}%)")


if __name__ == "__main__":
    main()
