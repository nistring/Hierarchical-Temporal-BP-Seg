"""CVAT XML ↔ COCO seed JSONs + per-frame PNG extraction for SUIT data.

Three subcommands:

* ``extract`` - Full pipeline: walk a CVAT annotation tree, crop every frame
  of each matching mp4 to the ultrasound viewport, write
  ``data/SUIT/images/<name>/image_*.png`` and
  ``data/SUIT/coco_annotations/<name>.json``. Bootstraps each split from a
  raw CVAT export.

* ``regenerate`` - Rebuild the ``annotations`` array of an existing seed JSON
  from the original CVAT XML, applying two fixes ``extract`` never did:
    1. Drop the trailing run of inside boxes whose
       ``(xtl, ytl, xbr, ybr)`` are bit-identical to the very last inside
       box (the upstream tracker stalled but the annotator never marked
       ``outside='1'``) - keep only the first frame of that run, the last
       actively-placed bbox.
    2. Trim the per-video ``images`` list to the new last-annotated frame
       so the dataloader doesn't serve the un-annotated tail as
       all-background. PNGs on disk are left alone.

* ``reextract-images`` - Rebuild PNGs deterministically using an existing
  seed JSON's ``(video_id, frame_id, file_name)`` mapping. Useful when the
  PNGs are deleted but the JSONs survive — running ``extract`` from
  scratch would shuffle image_id assignment because of ``os.listdir``
  ordering.

The ``crop()`` helper is also imported by ``data/manual_xml_to_coco.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
RAW_VID_DIR = ROOT / "data/raw/videos"
COCO_DIR = ROOT / "data/SUIT/coco_annotations"
IMG_ROOT = ROOT / "data/SUIT/images"

# split_name -> CVAT anno_root. Used by ``regenerate``.
SPLITS = {
    "sonosite_train": ROOT / "data/raw/anno/train",
    "sonosite_val":   ROOT / "data/raw/anno/val",
    "ge_train":       ROOT / "data/raw/anno/GE_train",
    "ge_val":         ROOT / "data/raw/anno/GE_val",
    "mindray_train":  ROOT / "data/raw/anno/mindray_train",
    "mindray_val":    ROOT / "data/raw/anno/mindray_val",
}

# --- Crop helper (largest-contour bbox of the ultrasound viewport) ---
THRESH_BINARY = 20
KERNEL_SIZE = 8

# Hard-coded viewport for one Sonosite video that the auto-crop misses;
# referenced by all three subcommands so they stay consistent.
ROI_OVERRIDE = {"00610090": (639, 130, 819, 747)}

LABELS = ["C5", "C6", "C7", "C8", "UT", "MT", "LT", "SSN", "AD", "PD"]
LBL2ID = {n: i + 1 for i, n in enumerate(LABELS)}


def crop(frame, roi=None):
    """Return ``(x, y, w, h)`` of the largest-contour bbox in ``frame``.

    If ``roi`` is given, the search is constrained to that ``(x, y, w, h)``
    sub-region and the result is reported in original-frame coordinates.
    """
    if roi is not None:
        frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    img = cv2.threshold(gray, THRESH_BINARY, 255, cv2.THRESH_TOZERO)[1]
    ret = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if roi is not None:
        x += roi[0]
        y += roi[1]
    return (x, y, w, h)


def _video_crop_bbox(video_basename, fallback_roi=None):
    """Compute the crop bbox for a raw video, applying the 00610090 override.

    ``video_basename`` excludes the ``.mp4`` suffix.
    Returns ``(x, y, w, h)`` or ``None`` if the video can't be opened.
    """
    if video_basename in ROI_OVERRIDE:
        return ROI_OVERRIDE[video_basename]
    cap = cv2.VideoCapture(str(RAW_VID_DIR / f"{video_basename}.mp4"))
    ok, first = cap.read()
    cap.release()
    if not ok:
        return None
    return crop(first, roi=fallback_roi)


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------
def cmd_extract(anno_root, out_name=None, roi=None):
    """Walk ``anno_root``, extract cropped frames + build a fresh COCO seed.

    Mirrors the original ``cvat_to_coco.preprocess`` behavior: image_ids
    are assigned by ``os.listdir`` order over ``anno_root`` subdirectories,
    so the resulting JSON is reproducible only on the same filesystem
    state. When PNGs already exist and you just want to refresh the COCO
    file, use ``regenerate`` instead.
    """
    derived = Path(anno_root).name
    anno_dir = out_name or derived
    is_vid_train_frame = "train" in anno_dir
    ann_file = COCO_DIR / f"{anno_dir}.json"
    save_dir = IMG_ROOT / anno_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(LABELS)]
    cat_list = [c["name"] for c in categories]
    videos, images, annotations = [], [], []
    img_id = anno_id = 1

    for vid_id, sub in enumerate(os.listdir(anno_root), start=1):
        tree = ET.parse(os.path.join(anno_root, sub, "annotations.xml"))
        root = tree.getroot()
        name = sub + ".mp4"
        first_frame_img_id = img_id
        print(name)

        videos.append({"id": vid_id, "name": name})

        # last frame with any inside box across all tracks
        max_frame_with_annotation = -1
        for track in root.findall("./track"):
            for box in track:
                attr = box.attrib
                if not bool(int(attr["outside"])):
                    max_frame_with_annotation = max(
                        max_frame_with_annotation, int(attr["frame"])
                    )

        bbox = None
        cap = cv2.VideoCapture(str(RAW_VID_DIR / name))
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frame_with_annotation >= 0 and frame_id > max_frame_with_annotation:
                break
            if bbox is None:
                bbox = ROI_OVERRIDE.get(sub) or crop(frame, roi=roi)
            file_name = f"image_{img_id}.png"
            cv2.imwrite(
                str(save_dir / file_name),
                frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]],
            )
            images.append({
                "file_name": file_name,
                "height": bbox[3], "width": bbox[2],
                "id": img_id, "video_id": vid_id, "frame_id": frame_id,
            })
            img_id += 1
            frame_id += 1
        cap.release()

        for track in root.findall("./track"):
            cat_name = track.attrib["label"].upper()
            for box in track:
                attr = box.attrib
                if bool(int(attr["outside"])):
                    continue
                width = float(attr["xbr"]) - float(attr["xtl"])
                height = float(attr["ybr"]) - float(attr["ytl"])
                annotations.append({
                    "id": anno_id,
                    "image_id": first_frame_img_id + int(attr["frame"]),
                    "video_id": vid_id,
                    "category_id": cat_list.index(cat_name) + 1,
                    "instance_id": 1,
                    "bbox": [
                        float(attr["xtl"]) - bbox[0],
                        float(attr["ytl"]) - bbox[1],
                        width, height,
                    ],
                    "area": width * height,
                    "occluded": bool(int(attr["occluded"])),
                    "truncated": False,
                    "iscrowd": False,
                    "is_vid_train_frame": is_vid_train_frame,
                    "visibility": 1.0,
                })
                anno_id += 1

    ann_file.parent.mkdir(parents=True, exist_ok=True)
    ann_file.write_text(json.dumps({
        "categories": categories,
        "videos": videos,
        "images": images,
        "annotations": annotations,
    }))
    print(f"wrote {ann_file}  ({len(images)} images, {len(annotations)} annotations)")


# ---------------------------------------------------------------------------
# Subcommand: regenerate
# ---------------------------------------------------------------------------
def cmd_regenerate(split):
    """Rebuild ``coco['annotations']`` and trim ``coco['images']`` for ``split``.

    Trailing-static-bbox runs (inside-only walk, ignoring intervening
    ``outside='1'`` markers) are dropped except for the first frame.
    Interior static-bbox runs are kept. ``images`` is trimmed per-video to
    the new last-annotated frame.
    """
    anno_root = SPLITS[split]
    coco_path = COCO_DIR / f"{split}.json"
    coco = json.loads(coco_path.read_text())

    vid_name_by_id = {v["id"]: v["name"].rsplit(".", 1)[0] for v in coco["videos"]}
    name_to_vid_id = {n: i for i, n in vid_name_by_id.items()}
    img_by_id = {im["id"]: im for im in coco["images"]}

    # Per-video: frame_id -> image_id
    images_by_vid: dict[int, dict[int, int]] = {}
    for im in coco["images"]:
        images_by_vid.setdefault(im["video_id"], {})[im["frame_id"]] = im["id"]

    # Per-video crop offset (cx, cy) — re-run crop() on the raw mp4 first
    # frame so we can convert raw bbox coordinates back into post-crop space.
    cx_cy_by_vid: dict[int, tuple[int, int]] = {}
    for vid_id, vname in vid_name_by_id.items():
        bbox = _video_crop_bbox(vname)
        if bbox is None:
            print(f"  WARN: cannot read {vname}.mp4 — skipping crop offset")
            cx_cy_by_vid[vid_id] = (0, 0)
        else:
            cx_cy_by_vid[vid_id] = (bbox[0], bbox[1])

    is_vid_train_frame = "train" in split
    new_annotations: list[dict] = []
    ann_id = 1
    n_skipped_orphan = 0
    n_outside = 0
    n_static_dup = 0

    for sub in sorted(anno_root.iterdir()):
        x = sub / "annotations.xml"
        if not x.exists():
            continue
        vname = sub.name
        vid_id = name_to_vid_id.get(vname)
        if vid_id is None:
            continue
        cx, cy = cx_cy_by_vid[vid_id]
        frame_to_img = images_by_vid.get(vid_id, {})

        for trk in ET.parse(x).getroot().findall("./track"):
            cat_name = trk.attrib["label"].upper()
            cat_id = LBL2ID.get(cat_name)
            if cat_id is None:
                continue
            track_boxes = sorted(
                trk.findall("box"),
                key=lambda b: int(b.attrib.get("frame", "0")),
            )
            # Trailing static-bbox run within the inside-only sub-sequence.
            inside_idxs = [i for i, b in enumerate(track_boxes)
                           if int(b.attrib.get("outside", "0")) == 0]
            trailing_run_pos: list[int] = []
            last_coords: tuple[float, float, float, float] | None = None
            for k in range(len(inside_idxs) - 1, -1, -1):
                a = track_boxes[inside_idxs[k]].attrib
                coords_k = (float(a["xtl"]), float(a["ytl"]),
                            float(a["xbr"]), float(a["ybr"]))
                if last_coords is None:
                    last_coords = coords_k
                    trailing_run_pos.append(k)
                elif coords_k == last_coords:
                    trailing_run_pos.append(k)
                else:
                    break
            trailing_run_pos.sort()
            drop_idxs = ({inside_idxs[k] for k in trailing_run_pos[1:]}
                         if len(trailing_run_pos) > 1 else set())

            for i, box in enumerate(track_boxes):
                a = box.attrib
                if int(a.get("outside", "0")) == 1:
                    n_outside += 1
                    continue
                if i in drop_idxs:
                    n_static_dup += 1
                    continue
                xtl_raw = float(a["xtl"])
                ytl_raw = float(a["ytl"])
                xbr_raw = float(a["xbr"])
                ybr_raw = float(a["ybr"])
                f = int(a["frame"])
                img_id = frame_to_img.get(f)
                if img_id is None:
                    n_skipped_orphan += 1
                    continue
                xtl = xtl_raw - cx
                ytl = ytl_raw - cy
                w = xbr_raw - xtl_raw
                h = ybr_raw - ytl_raw
                new_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "video_id": vid_id,
                    "category_id": cat_id,
                    "instance_id": 1,
                    "bbox": [xtl, ytl, w, h],
                    "area": w * h,
                    "occluded": bool(int(a.get("occluded", "0"))),
                    "truncated": False,
                    "iscrowd": False,
                    "is_vid_train_frame": is_vid_train_frame,
                    "visibility": 1.0,
                })
                ann_id += 1

    coco["annotations"] = new_annotations

    # Trim ``images`` per video to the new last-annotated frame.
    max_frame_by_vid: dict[int, int] = {}
    for ann in new_annotations:
        im = img_by_id.get(ann["image_id"])
        if im is None:
            continue
        if im["frame_id"] > max_frame_by_vid.get(im["video_id"], -1):
            max_frame_by_vid[im["video_id"]] = im["frame_id"]
    n_images_before = len(coco["images"])
    coco["images"] = [
        im for im in coco["images"]
        if im["frame_id"] <= max_frame_by_vid.get(im["video_id"], -1)
    ]
    n_images_dropped = n_images_before - len(coco["images"])

    coco_path.write_text(json.dumps(coco))
    print(f"  {split}: {len(new_annotations)} annotations  "
          f"(outside skipped: {n_outside}, static-dup skipped: {n_static_dup}, "
          f"orphan frames skipped: {n_skipped_orphan}, "
          f"images trimmed: {n_images_dropped} / {n_images_before})")


# ---------------------------------------------------------------------------
# Subcommand: reextract-images
# ---------------------------------------------------------------------------
def cmd_reextract_images(split):
    """Re-write the PNGs for ``split`` deterministically from the seed JSON.

    For each video, recover the crop bbox via :func:`_video_crop_bbox` and
    sequentially read frames; whenever ``frame_id`` is listed in the JSON
    we save the crop with the JSON-recorded ``file_name``.
    """
    coco = json.loads((COCO_DIR / f"{split}.json").read_text())
    save_dir = IMG_ROOT / split
    save_dir.mkdir(parents=True, exist_ok=True)

    vid2name = {v["id"]: v["name"].rsplit(".", 1)[0] for v in coco["videos"]}
    by_vid: dict[int, list[dict]] = {}
    for im in coco["images"]:
        by_vid.setdefault(im["video_id"], []).append(im)

    n_total = 0
    n_videos = 0
    for vid_id in sorted(by_vid.keys()):
        vname = vid2name[vid_id]
        raw_mp4 = RAW_VID_DIR / f"{vname}.mp4"
        if not raw_mp4.exists():
            print(f"  ! missing raw video: {raw_mp4}")
            continue
        bbox = _video_crop_bbox(vname)
        if bbox is None:
            print(f"  ! cannot read first frame of {raw_mp4}")
            continue
        cx, cy, cw, ch = bbox

        ims_by_fid = {im["frame_id"]: im for im in by_vid[vid_id]}
        max_fid = max(ims_by_fid.keys())

        cap = cv2.VideoCapture(str(raw_mp4))
        n_video = 0
        for fid in range(max_fid + 1):
            ok, frame = cap.read()
            if not ok:
                break
            im = ims_by_fid.get(fid)
            if im is None:
                continue
            cropped = frame[cy:cy + ch, cx:cx + cw]
            if (cropped.shape[1], cropped.shape[0]) != (im["width"], im["height"]):
                print(f"  ! crop size mismatch on {vname} frame {fid}: "
                      f"got {cropped.shape[1]}x{cropped.shape[0]} "
                      f"want {im['width']}x{im['height']}")
            cv2.imwrite(str(save_dir / im["file_name"]), cropped)
            n_video += 1
        cap.release()
        if n_video != len(by_vid[vid_id]):
            print(f"  ! {vname}: wrote {n_video} / {len(by_vid[vid_id])} expected")
        n_total += n_video
        n_videos += 1

    print(f"  {split}: {n_total} images written across {n_videos} videos")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="CVAT XML -> COCO seed + cropped PNGs")
    pe.add_argument("--anno-root", required=True,
                    help="e.g. data/raw/anno/val  or  data/raw/anno/GE_train")
    pe.add_argument("--out-name", default=None,
                    help="Output basename (no .json). Default: last path "
                         "segment of --anno-root.")
    pe.add_argument("--roi", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="Manual ROI (x y w h) passed to crop(); use for GE 1080p videos.")

    pr = sub.add_parser(
        "regenerate",
        help="Rebuild annotations + trim images of existing seed JSONs",
    )
    pr.add_argument("splits", nargs="*",
                    help=f"Split names; default = all of {sorted(SPLITS)}.")

    px = sub.add_parser(
        "reextract-images",
        help="Re-write PNGs for existing seed JSONs (deterministic).",
    )
    px.add_argument("splits", nargs="*",
                    help=f"Split names; default = all of {sorted(SPLITS)}.")

    return p


def main():
    args = _build_parser().parse_args()
    if args.cmd == "extract":
        cmd_extract(
            args.anno_root,
            out_name=args.out_name,
            roi=tuple(args.roi) if args.roi else None,
        )
        return
    targets = args.splits or list(SPLITS)
    for split in targets:
        if split not in SPLITS:
            print(f"unknown split: {split}")
            continue
        if args.cmd == "regenerate":
            print(f"Regenerating {split}...")
            cmd_regenerate(split)
        elif args.cmd == "reextract-images":
            print(f"Re-extracting {split}...")
            cmd_reextract_images(split)


if __name__ == "__main__":
    main()
