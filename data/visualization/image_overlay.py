"""Image-level annotation overlays from COCO segmentation files.

Two modes:

  per-frame
      One overlay PNG per annotated image, with semi-transparent fills,
      outlines, and a banner that names the source video, file, and the
      classes drawn.

  grid
      A matplotlib grid figure: N videos x M frames per video, each
      panel showing the val frame with polygons.

Examples:
    PYTHONPATH=. python -m data.visualization.image_overlay per-frame \\
        --coco data/SUIT/coco_annotations/sonosite_val_manual.json \\
        --img-dir data/SUIT/images/sonosite_val \\
        --out out/manual_overlays

    PYTHONPATH=. python -m data.visualization.image_overlay grid \\
        --coco data/SUIT/coco_annotations/ge_train_sam2.json \\
        --img-dir data/SUIT/images/ge_train \\
        --out out/figures/autolabel_inspect/ge.png \\
        --videos 4 --frames-per-video 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data.visualization._overlay import (
    CLASS_NAMES, CLASS_RGB_FLOAT,
    banner, classes_summary, coco_anns_to_polys, overlay_polygons,
)


def _load_coco(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _index(coco: dict):
    """Return (img_by_id, vid_by_id, by_video, by_image_anns)."""
    img_by_id = {im["id"]: im for im in coco["images"]}
    vid_by_id = {v["id"]: v.get("name", str(v["id"])) for v in coco.get("videos", [])}
    by_video: dict[int, list[dict]] = defaultdict(list)
    for im in coco["images"]:
        v = im.get("video_id")
        if v is not None:
            by_video[v].append(im)
    for v in by_video.values():
        v.sort(key=lambda x: x["file_name"])
    by_image_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image_anns[ann["image_id"]].append(ann)
    return img_by_id, vid_by_id, by_video, by_image_anns


# ---------- per-frame mode ----------------------------------------------------

def render_per_frame(coco_path: Path, img_dir: Path, out_dir: Path,
                     alpha: float = 0.45) -> int:
    """Write one overlay PNG per annotated image. Returns count of PNGs."""
    coco = _load_coco(coco_path)
    img_by_id, vid_by_id, _, by_image_anns = _index(coco)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for img_id in sorted(by_image_anns):
        polys = coco_anns_to_polys(by_image_anns[img_id])
        if not polys:
            continue
        info = img_by_id[img_id]
        path = Path(img_dir) / info["file_name"]
        if not path.exists():
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        out = overlay_polygons(bgr, polys, alpha=alpha)
        vname = vid_by_id.get(info.get("video_id"), "?")
        banner(out, f"{vname}  {info['file_name']}  classes: {classes_summary(polys)}")
        sub = out_dir / Path(vname).stem
        sub.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sub / info["file_name"]), out)
        n_written += 1
    print(f"Wrote {n_written} overlay PNGs to {out_dir}")
    return n_written


# ---------- grid mode ---------------------------------------------------------

def _evenly_sample(seq, n):
    if len(seq) <= n:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in idx]


def render_grid(coco_path: Path, img_dir: Path, out_path: Path,
                n_videos: int = 4, n_frames: int = 5,
                seed: int = 0, alpha: float = 0.45,
                title: str | None = None) -> Path:
    """Render an N_videos x N_frames matplotlib figure with polygon overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from PIL import Image

    coco = _load_coco(coco_path)
    _, vid_by_id, by_video, by_image_anns = _index(coco)

    rng = random.Random(seed)
    vids = sorted(by_video.keys())
    rng.shuffle(vids)
    vids = vids[:n_videos]

    fig, axes = plt.subplots(
        n_videos, n_frames,
        figsize=(2.6 * n_frames, 2.6 * n_videos),
        gridspec_kw={"wspace": 0.02, "hspace": 0.18},
    )
    if n_videos == 1:
        axes = axes[None, :]

    for row, vid_id in enumerate(vids):
        imgs = _evenly_sample(by_video[vid_id], n_frames)
        for col, info in enumerate(imgs):
            ax = axes[row, col]
            path = Path(img_dir) / info["file_name"]
            if not path.exists():
                ax.set_axis_off()
                ax.set_title("(missing)", fontsize=8)
                continue
            arr = np.array(Image.open(path).convert("L"))
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)

            anns = by_image_anns.get(info["id"], [])
            polys = coco_anns_to_polys(anns)
            by_class: dict[int, list[np.ndarray]] = defaultdict(list)
            for cls, pts in polys:
                by_class[cls].append(pts.astype(np.float32))
            for cls, ps in by_class.items():
                color = CLASS_RGB_FLOAT[cls - 1]
                patches = [Polygon(p, closed=True) for p in ps]
                coll = PatchCollection(
                    patches, facecolor=color, edgecolor=color,
                    linewidths=1.0, alpha=alpha,
                )
                ax.add_collection(coll)

            ax.set_xticks([]); ax.set_yticks([])
            classes_drawn = sorted(by_class.keys())
            label = ", ".join(CLASS_NAMES[c] for c in classes_drawn) or "no mask"
            ax.set_xlabel(label, fontsize=7)
            if col == 0:
                ax.set_ylabel(vid_by_id.get(vid_id, str(vid_id)),
                              fontsize=8, rotation=90, labelpad=4)

    fig.suptitle(
        title or f"{Path(coco_path).stem}  —  {n_videos} videos × {n_frames} frames",
        fontsize=12, y=0.995,
    )
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="",
                   markerfacecolor=CLASS_RGB_FLOAT[i - 1],
                   markeredgecolor="black", markersize=8, label=name)
        for i, name in enumerate(CLASS_NAMES[1:], start=1)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote grid figure to {out_path}")
    return out_path


# ---------- CLI ---------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(description="Render annotation overlays on images.")
    sub = p.add_subparsers(dest="mode", required=True)

    pf = sub.add_parser("per-frame", help="One PNG per annotated image.")
    pf.add_argument("--coco", required=True, type=Path)
    pf.add_argument("--img-dir", required=True, type=Path)
    pf.add_argument("--out", required=True, type=Path)
    pf.add_argument("--alpha", type=float, default=0.45)

    g = sub.add_parser("grid", help="Multi-frame matplotlib grid figure.")
    g.add_argument("--coco", required=True, type=Path)
    g.add_argument("--img-dir", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--videos", type=int, default=4)
    g.add_argument("--frames-per-video", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--alpha", type=float, default=0.45)
    g.add_argument("--title", type=str, default=None)
    return p


def main():
    args = _build_parser().parse_args()
    if args.mode == "per-frame":
        render_per_frame(args.coco, args.img_dir, args.out, alpha=args.alpha)
    elif args.mode == "grid":
        render_grid(
            args.coco, args.img_dir, args.out,
            n_videos=args.videos, n_frames=args.frames_per_video,
            seed=args.seed, alpha=args.alpha, title=args.title,
        )


if __name__ == "__main__":
    main()
