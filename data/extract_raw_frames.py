"""Extract cropped + resized PNGs from unlabeled raw videos for SSL pseudo-labeling.

For each raw video in data/raw/videos/<domain>/ (domain = GE | mindray) whose
patient ID is NOT already in the labeled splits (`<dom>_train` / `<dom>_val`),
read every frame at the native fps, crop to the ultrasound viewport using the
same `crop()` helper used by the supervised pipeline, resize to image_size
(default 416x416), and save as PNG under `data/SUIT/images/<dom>_pseudo/`.

Also build a seed COCO at `data/SUIT/coco_annotations/<dom>_pseudo_seed.json`
with `images` + empty `annotations`. `pseudo_label.py` then fills in the masks.

Usage:
    PYTHONPATH=. python data/extract_raw_frames.py --domain ge
    PYTHONPATH=. python data/extract_raw_frames.py --domain mindray
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from data.cvat_to_coco import crop, ROI_OVERRIDE, LABELS

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/videos"
ANNO_DIR = ROOT / "data/raw/anno"
IMG_ROOT = ROOT / "data/SUIT/images"
COCO_DIR = ROOT / "data/SUIT/coco_annotations"

DOMAIN_RAW = {"ge": "GE", "mindray": "mindray"}
LABELED_SPLITS = {
    "ge":      ["GE_train",      "GE_val"],
    "mindray": ["mindray_train", "mindray_val"],
}


def labeled_patient_ids(domain: str) -> set[str]:
    ids = set()
    for split in LABELED_SPLITS[domain]:
        d = ANNO_DIR / split
        if not d.exists():
            continue
        for sub in d.iterdir():
            if sub.is_dir():
                ids.add(sub.name)
    return ids


def extract(domain: str, image_size: int = 416, stride: int = 1):
    raw_dir = RAW_DIR / DOMAIN_RAW[domain]
    out_img_dir = IMG_ROOT / f"{domain}_pseudo"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_coco = COCO_DIR / f"{domain}_pseudo_seed.json"

    skip_ids = labeled_patient_ids(domain)
    raw_videos = sorted(p for p in raw_dir.iterdir() if p.suffix == ".mp4")
    keep = [p for p in raw_videos if p.stem not in skip_ids]
    skipped = [p.stem for p in raw_videos if p.stem in skip_ids]
    print(f"[{domain}] {len(raw_videos)} raw, {len(skip_ids)} labeled IDs, "
          f"{len(keep)} truly-new, {len(skipped)} skipped (overlap with labeled)")

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(LABELS)]
    videos, images = [], []
    img_id = 1

    for vid_id, mp4 in enumerate(keep, start=1):
        cap = cv2.VideoCapture(str(mp4))
        ok, first = cap.read()
        if not ok:
            print(f"  WARN: cannot open {mp4}")
            cap.release()
            continue
        bbox = ROI_OVERRIDE.get(mp4.stem) or crop(first)
        cx, cy, cw, ch = bbox
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        videos.append({"id": vid_id, "name": mp4.name})
        first_img_id = img_id
        n_kept = 0
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % stride == 0:
                cropped = frame[cy:cy + ch, cx:cx + cw]
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (image_size, image_size),
                                     interpolation=cv2.INTER_AREA)
                file_name = f"image_{img_id}.png"
                cv2.imwrite(str(out_img_dir / file_name), resized)
                images.append({
                    "file_name": file_name,
                    "height": image_size, "width": image_size,
                    "id": img_id, "video_id": vid_id, "frame_id": frame_id,
                })
                img_id += 1
                n_kept += 1
            frame_id += 1
        cap.release()
        print(f"  [{vid_id:3d}/{len(keep)}] {mp4.stem}: {n_kept} frames "
              f"(crop {cw}x{ch} -> {image_size}x{image_size})")

    out_coco.parent.mkdir(parents=True, exist_ok=True)
    out_coco.write_text(json.dumps({
        "categories": categories,
        "videos": videos,
        "images": images,
        "annotations": [],
    }))
    print(f"wrote {out_coco}  ({len(images)} images, {len(videos)} videos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAIN_RAW), required=True)
    ap.add_argument("--image-size", type=int, default=416)
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every Nth frame. 1 = native fps, 2 = half fps.")
    args = ap.parse_args()
    extract(args.domain, image_size=args.image_size, stride=args.stride)


if __name__ == "__main__":
    main()
