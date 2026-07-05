"""Video-level annotation overlays from COCO segmentation files.

Walks a COCO file, groups images by video, and renders one mp4 per video
with the polygons burned on as semi-transparent fills + outlines + a banner
naming the video, frame number, and classes drawn.

Convenience: --preset selects a named (coco, img_dir, output-subdir) triple
for the SUIT autolabel splits we render most often. Pass --coco/--img-dir
explicitly for ad hoc inputs (e.g. the manual COCO).

Examples:
    # render the four SAM2 autolabel splits in parallel
    PYTHONPATH=. python -m data.visualization.video_overlay \\
        --preset ge_train ge_val mindray_train mindray_val \\
        --workers 8 --fps 20

    # render manual ground-truth overlays
    PYTHONPATH=. python -m data.visualization.video_overlay \\
        --coco data/SUIT/coco_annotations/sonosite_val_manual.json \\
        --img-dir data/SUIT/images/sonosite_val \\
        --out out/manual_video_overlays \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

from data.visualization._overlay import (
    banner, classes_summary, coco_anns_to_polys, overlay_polygons,
)


ROOT = Path(__file__).resolve().parents[2]

# Preset: (coco_relative_path, img_dir_relative_path, output_subdir_name)
PRESETS = {
    "sonosite_train": ("data/SUIT/coco_annotations/sonosite_train_sam2.json", "data/SUIT/images/sonosite_train", "sonosite_train"),
    "sonosite_val":   ("data/SUIT/coco_annotations/sonosite_val_sam2.json",   "data/SUIT/images/sonosite_val",   "sonosite_val"),
    "ge_train":       ("data/SUIT/coco_annotations/ge_train_sam2.json",       "data/SUIT/images/ge_train",       "ge_train"),
    "ge_val":         ("data/SUIT/coco_annotations/ge_val_sam2.json",         "data/SUIT/images/ge_val",         "ge_val"),
    "mindray_train":  ("data/SUIT/coco_annotations/mindray_train_sam2.json",  "data/SUIT/images/mindray_train",  "mindray_train"),
    "mindray_val":    ("data/SUIT/coco_annotations/mindray_val_sam2.json",    "data/SUIT/images/mindray_val",    "mindray_val"),
}
DEFAULT_OUT_ROOT = ROOT / "out/autolabel_videos"


def _index_by_video(coco: dict):
    vid2name = {v["id"]: v.get("name", str(v["id"])) for v in coco.get("videos", [])}
    vid2imgs: dict[int, list[dict]] = defaultdict(list)
    for img in coco["images"]:
        vid = img.get("video_id")
        if vid is not None:
            vid2imgs[vid].append(img)
    for v in vid2imgs.values():
        # Sort by frame_id, NOT file_name. file_name embeds the global
        # image_id counter (e.g. image_1.png .. image_999.png), so a
        # lexicographic sort interleaves frames every time the digit count
        # changes, splicing image_10.png between image_1.png and image_2.png.
        v.sort(key=lambda x: x["frame_id"])
    img2anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        img2anns[ann["image_id"]].append(ann)
    return vid2name, vid2imgs, img2anns


def _render_video(video_name: str,
                  imgs: list[dict],
                  img2anns: dict[int, list[dict]],
                  img_dir: Path,
                  out_path: Path,
                  fps: int,
                  alpha: float,
                  show_progress: bool = False) -> None:
    if not imgs:
        return
    first_path = img_dir / imgs[0]["file_name"]
    sample = cv2.imread(str(first_path))
    if sample is None:
        print(f"  ! cannot read {first_path}, skipping {video_name}")
        return
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    iterator = tqdm(imgs, desc=video_name, leave=False) if show_progress else imgs
    for i, info in enumerate(iterator):
        bgr = cv2.imread(str(img_dir / info["file_name"]))
        if bgr is None:
            continue
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h))
        polys = coco_anns_to_polys(img2anns.get(info["id"], []))
        out = overlay_polygons(bgr, polys, alpha=alpha)
        banner(
            out,
            f"{video_name}  frame {i + 1}/{len(imgs)}  "
            f"classes: {classes_summary(polys)}",
        )
        writer.write(out)
    writer.release()


def _render_one(args):
    """Worker entry point — top-level so it pickles for ProcessPoolExecutor."""
    (vname, imgs, anns_for_imgs, img_dir, out_path, fps, alpha) = args
    img2anns = defaultdict(list, anns_for_imgs)
    t0 = time.time()
    _render_video(vname, imgs, img2anns, img_dir, out_path, fps, alpha)
    sz = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
    return (vname, len(imgs), sz, time.time() - t0)


def render_split(coco_path: Path, img_dir: Path, out_dir: Path,
                 fps: int = 20, alpha: float = 0.45,
                 only_video: str | None = None,
                 workers: int = 8) -> int:
    """Render one mp4 per video listed in ``coco_path``."""
    coco = json.loads(Path(coco_path).read_text())
    vid2name, vid2imgs, img2anns = _index_by_video(coco)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for vid_id in sorted(vid2imgs.keys()):
        vname = vid2name.get(vid_id, str(vid_id))
        if only_video and only_video not in vname:
            continue
        imgs = vid2imgs[vid_id]
        out_path = out_dir / f"{Path(vname).stem}.mp4"
        ids_needed = {im["id"] for im in imgs}
        anns_subset = {iid: img2anns[iid] for iid in ids_needed if iid in img2anns}
        jobs.append((vname, imgs, anns_subset, Path(img_dir), out_path, fps, alpha))

    n_done = 0
    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            vname, n, sz, dt = _render_one(j)
            print(f"  wrote {(out_dir / Path(vname).stem)}.mp4 "
                  f"({sz:.1f} MB, {dt:.1f}s, {n} frames)")
            n_done += 1
        return n_done

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_render_one, j): j[0] for j in jobs}
        for fut in as_completed(futures):
            vname, n, sz, dt = fut.result()
            print(f"  wrote {(out_dir / Path(vname).stem)}.mp4 "
                  f"({sz:.1f} MB, {dt:.1f}s, {n} frames)")
            n_done += 1
    return n_done


def _build_parser():
    p = argparse.ArgumentParser(description="Render polygon-overlay videos from COCO.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", nargs="+", choices=list(PRESETS.keys()),
                     help="Named SUIT autolabel splits.")
    src.add_argument("--coco", type=Path, help="COCO json (use with --img-dir).")
    p.add_argument("--img-dir", type=Path, help="Directory holding the val PNGs.")
    p.add_argument("--out", type=Path,
                   help="Output directory. Defaults to out/autolabel_videos/<preset>.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--only-video", default=None,
                   help="Substring filter on video name (debug).")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel video workers (1 = serial).")
    return p


def main():
    args = _build_parser().parse_args()
    if args.preset:
        for split in args.preset:
            coco_rel, img_rel, sub = PRESETS[split]
            coco_p = ROOT / coco_rel
            img_p = ROOT / img_rel
            out_p = args.out or (DEFAULT_OUT_ROOT / sub)
            print(f"\n=== {split} ===")
            n = render_split(coco_p, img_p, out_p, fps=args.fps, alpha=args.alpha,
                             only_video=args.only_video, workers=args.workers)
            print(f"  {n} video(s) rendered for {split}")
        print(f"\nAll outputs under: {args.out or DEFAULT_OUT_ROOT}")
    else:
        if args.img_dir is None or args.out is None:
            raise SystemExit("--coco requires --img-dir and --out")
        n = render_split(args.coco, args.img_dir, args.out,
                         fps=args.fps, alpha=args.alpha,
                         only_video=args.only_video, workers=args.workers)
        print(f"\n{n} video(s) rendered to {args.out}")


if __name__ == "__main__":
    main()
