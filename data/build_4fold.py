"""Build 4-fold cross-validation splits for the R1.3 target-vendor experiments.

Pools the 12 + 12 labeled videos per vendor (`{vendor}_train_{labels}.json`
union `{vendor}_val_{labels}.json`) into 24, then partitions deterministically
into 4 folds of 6 videos each. For fold k, the train COCO contains the other
18 videos and the val COCO contains the 6 videos in fold k.

Per-frame filenames in the existing COCOs are flat (`image_NNN.png`) and
collide between `ge_train/` and `ge_val/` image directories. The fold COCO
prefixes every `file_name` with its source subdir (`ge_train/...` or
`ge_val/...`) so a single `data_path: data/SUIT/images/` resolves everything.

Fold assignment is deterministic: videos are sorted by `name`; the i-th sorted
video goes to fold `i % 4`. No RNG, no shuffle — reproducible without seed.

Usage:
    PYTHONPATH=. python data/build_4fold.py \\
        --vendors ge mindray --labels sam2 ultrasam
    # writes data/SUIT/coco_annotations/{vendor}_fold{0..3}_{train,val}_{labels}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "SUIT" / "coco_annotations"
NUM_FOLDS = 4


def _load(coco_path: Path) -> dict:
    return json.loads(coco_path.read_text())


def _prefix_filenames(coco: dict, prefix: str) -> dict:
    """Return a shallow-copied COCO with every image file_name prefixed."""
    new_imgs = []
    for im in coco["images"]:
        new_im = dict(im)
        new_im["file_name"] = f"{prefix}/{im['file_name']}"
        new_imgs.append(new_im)
    return {**coco, "images": new_imgs}


def _pool(vendor: str, labels: str) -> tuple[dict, dict[int, str]]:
    """Concatenate `{vendor}_train_{labels}.json` and `{vendor}_val_{labels}.json`.

    Returns (pooled_coco, vid_id -> source_subdir).
    Video / image / annotation IDs are re-numbered to a fresh contiguous range
    so train+val IDs don't collide.
    """
    train = _prefix_filenames(_load(COCO_DIR / f"{vendor}_train_{labels}.json"),
                              f"{vendor}_train")
    val = _prefix_filenames(_load(COCO_DIR / f"{vendor}_val_{labels}.json"),
                            f"{vendor}_val")

    new_videos, new_images, new_anns = [], [], []
    vid_src: dict[int, str] = {}
    next_vid, next_img, next_ann = 1, 1, 1

    for src_tag, coco in (("ge_train" if vendor == "ge" else "mindray_train", train),
                          ("ge_val"   if vendor == "ge" else "mindray_val",   val)):
        vid_remap, img_remap = {}, {}
        for v in coco["videos"]:
            vid_remap[v["id"]] = next_vid
            new_videos.append({**v, "id": next_vid})
            vid_src[next_vid] = src_tag
            next_vid += 1
        for im in coco["images"]:
            img_remap[im["id"]] = next_img
            ni = dict(im)
            ni["id"] = next_img
            ni["video_id"] = vid_remap[im["video_id"]]
            new_images.append(ni)
            next_img += 1
        for a in coco["annotations"]:
            na = dict(a)
            na["id"] = next_ann
            na["image_id"] = img_remap[a["image_id"]]
            na["video_id"] = vid_remap[a["video_id"]]
            new_anns.append(na)
            next_ann += 1

    pooled = {
        "categories": train.get("categories") or val.get("categories"),
        "videos": new_videos,
        "images": new_images,
        "annotations": new_anns,
    }
    return pooled, vid_src


def _fold_assignment(videos: list[dict]) -> dict[int, int]:
    """Sorted-by-name, stride-4 assignment.

    Sorted index i -> fold (i % 4). Deterministic, no shuffle.
    """
    sorted_vids = sorted(videos, key=lambda v: v["name"])
    return {v["id"]: i % NUM_FOLDS for i, v in enumerate(sorted_vids)}


def _split_by_fold(pooled: dict, vid_fold: dict[int, int], fold: int):
    """Return (train_coco, val_coco) for one fold k.

    train: videos with vid_fold != k (3 folds, ~18 videos)
    val:   videos with vid_fold == k (1 fold,  ~6 videos)
    """
    train_vids = {v["id"] for v in pooled["videos"] if vid_fold[v["id"]] != fold}
    val_vids   = {v["id"] for v in pooled["videos"] if vid_fold[v["id"]] == fold}

    def slice_to(vids: set[int]) -> dict:
        return {
            "categories": pooled["categories"],
            "videos":      [v  for v  in pooled["videos"]      if v["id"]       in vids],
            "images":      [im for im in pooled["images"]      if im["video_id"] in vids],
            "annotations": [a  for a  in pooled["annotations"] if a["video_id"] in vids],
        }
    return slice_to(train_vids), slice_to(val_vids)


def build_for(vendor: str, labels: str, write: bool) -> dict:
    pooled, _ = _pool(vendor, labels)
    vid_fold = _fold_assignment(pooled["videos"])

    summary = {"vendor": vendor, "labels": labels, "folds": []}
    for k in range(NUM_FOLDS):
        train, val = _split_by_fold(pooled, vid_fold, k)
        out_train = COCO_DIR / f"{vendor}_fold{k}_train_{labels}.json"
        out_val   = COCO_DIR / f"{vendor}_fold{k}_val_{labels}.json"
        if write:
            out_train.write_text(json.dumps(train))
            out_val.write_text(json.dumps(val))
        summary["folds"].append({
            "k": k,
            "train_videos": len(train["videos"]),
            "train_frames": len(train["images"]),
            "train_anns":   len(train["annotations"]),
            "val_videos":   len(val["videos"]),
            "val_frames":   len(val["images"]),
            "val_anns":     len(val["annotations"]),
            "val_names":    sorted(v["name"] for v in val["videos"]),
            "train_path":   str(out_train.relative_to(ROOT)),
            "val_path":     str(out_val.relative_to(ROOT)),
        })
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vendors", nargs="+", default=["ge", "mindray"],
                    choices=["ge", "mindray"])
    ap.add_argument("--labels", nargs="+", default=["sam2", "ultrasam"],
                    choices=["sam2", "ultrasam"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print the split but don't write files.")
    args = ap.parse_args()

    for vendor in args.vendors:
        for labels in args.labels:
            s = build_for(vendor, labels, write=not args.dry_run)
            print(f"\n=== {vendor} / {labels} ===")
            for f in s["folds"]:
                print(f"  fold {f['k']}: "
                      f"train {f['train_videos']:2d}v/{f['train_frames']:5d}f/{f['train_anns']:5d}a  "
                      f"val {f['val_videos']:2d}v/{f['val_frames']:5d}f/{f['val_anns']:5d}a")
                print(f"           val_videos = {f['val_names']}")


if __name__ == "__main__":
    main()
