"""Build a per-fold SSL training COCO by merging this fold's labeled train data
with vendor-level pseudo-labels (round 1 or round 2).

Output file_names are all rooted at ``data/SUIT/images/`` — labeled videos use
the existing ``{vendor}_{train,val}/`` prefix (already encoded by
``build_4fold.py``), and pseudo entries get a ``{vendor}_pseudo/`` prefix
prepended here. The resulting config can use ``data_path: data/SUIT/images/``
without any new symlink trees.

The pseudo-label set is shared across folds (the unlabeled video pool is
disjoint from labeled). The teacher that produced the pseudo-labels may have
seen some videos that are in this fold's val partition, but the student only
trains on (fold's labeled train ∪ pseudo of unlabeled pool) — no test-fold
images enter student training.

Usage:
    PYTHONPATH=. python data/build_4fold_ssl.py \\
        --vendor ge --fold 0 --labels ultrasam --round 1 \\
        --label-oversample 3

Writes: data/SUIT/coco_annotations/{vendor}_fold{k}_ssl_{labels}_train.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "SUIT" / "coco_annotations"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _prefix(coco: dict, prefix: str) -> dict:
    new_imgs = []
    for im in coco["images"]:
        ni = dict(im)
        ni["file_name"] = f"{prefix}/{im['file_name']}"
        new_imgs.append(ni)
    return {**coco, "images": new_imgs}


def _relabel(coco: dict, vid_off: int, img_off: int, ann_off: int):
    """Reassign IDs into a fresh contiguous range starting at the offsets."""
    vid_map, img_map = {}, {}
    new_videos, new_images, new_anns = [], [], []
    for i, v in enumerate(coco.get("videos", [])):
        vid_map[v["id"]] = vid_off + i
        new_videos.append({**v, "id": vid_off + i})
    for i, im in enumerate(coco["images"]):
        img_map[im["id"]] = img_off + i
        ni = dict(im)
        ni["id"] = img_off + i
        ni["video_id"] = vid_map[im["video_id"]]
        new_images.append(ni)
    for i, a in enumerate(coco["annotations"]):
        na = dict(a)
        na["id"] = ann_off + i
        na["image_id"] = img_map[a["image_id"]]
        if "video_id" in a and a["video_id"] in vid_map:
            na["video_id"] = vid_map[a["video_id"]]
        new_anns.append(na)
    return new_videos, new_images, new_anns


def build(vendor: str, fold: int, labels: str, round_: int,
          oversample: int, drop_short: int):
    labeled_path = COCO_DIR / f"{vendor}_fold{fold}_train_{labels}.json"
    if not labeled_path.exists():
        raise SystemExit(
            f"missing fold COCO: {labeled_path}. Run data/build_4fold.py first."
        )

    pseudo_name = f"{vendor}_pseudo.json" if round_ == 1 else f"{vendor}_pseudo_r2.json"
    pseudo_path = COCO_DIR / pseudo_name
    if not pseudo_path.exists():
        raise SystemExit(
            f"missing pseudo-label COCO: {pseudo_path}. "
            f"Generate it via data/pseudo_label.py (round {round_})."
        )

    labeled = _load(labeled_path)
    pseudo = _prefix(_load(pseudo_path), f"{vendor}_pseudo")

    # Drop short pseudo videos that would break the DataLoader's sequence_length collate.
    if drop_short > 1:
        counts: dict[int, int] = {}
        for im in pseudo["images"]:
            counts[im["video_id"]] = counts.get(im["video_id"], 0) + 1
        keep = {v for v, n in counts.items() if n >= drop_short}
        if len(keep) != len(counts):
            n_drop = len(counts) - len(keep)
            pseudo = {
                **pseudo,
                "videos": [v for v in pseudo.get("videos", []) if v["id"] in keep],
                "images": [im for im in pseudo["images"] if im["video_id"] in keep],
                "annotations": [a for a in pseudo["annotations"]
                                if a.get("video_id") in keep],
            }
            print(f"  dropped {n_drop} short pseudo video(s) (<{drop_short} frames)")

    videos, images, annotations = [], [], []
    vid_off = img_off = ann_off = 1
    for _ in range(oversample):
        v, i, a = _relabel(labeled, vid_off, img_off, ann_off)
        videos.extend(v); images.extend(i); annotations.extend(a)
        vid_off += len(v); img_off += len(i); ann_off += len(a)

    v, i, a = _relabel(pseudo, vid_off, img_off, ann_off)
    videos.extend(v); images.extend(i); annotations.extend(a)

    out_path = COCO_DIR / f"{vendor}_fold{fold}_ssl_{labels}_train.json"
    out_path.write_text(json.dumps({
        "categories": labeled.get("categories") or pseudo.get("categories"),
        "videos": videos,
        "images": images,
        "annotations": annotations,
    }))

    print(f"[{vendor} fold{fold} {labels} r{round_}] "
          f"labeled x{oversample}: {oversample*len(labeled['videos'])}v / "
          f"{oversample*len(labeled['images'])}f / "
          f"{oversample*len(labeled['annotations'])}a")
    print(f"  pseudo: {len(pseudo['videos'])}v / "
          f"{len(pseudo['images'])}f / {len(pseudo['annotations'])}a")
    print(f"  combined: {len(videos)}v / {len(images)}f / {len(annotations)}a")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vendor", required=True, choices=["ge", "mindray"])
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3])
    ap.add_argument("--labels", required=True, choices=["sam2", "ultrasam"])
    ap.add_argument("--round", dest="round_", type=int, required=True, choices=[1, 2])
    ap.add_argument("--label-oversample", type=int, default=3)
    ap.add_argument("--drop-short", type=int, default=50,
                    help="Drop pseudo videos with fewer than this many frames "
                         "(below sequence_length the DataLoader batch breaks).")
    args = ap.parse_args()
    build(args.vendor, args.fold, args.labels, args.round_,
          args.label_oversample, args.drop_short)


if __name__ == "__main__":
    main()
