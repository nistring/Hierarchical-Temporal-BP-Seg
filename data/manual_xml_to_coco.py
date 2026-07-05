"""Convert data/raw/anno/manual_annotations.xml -> COCO segmentation JSON.

Output: data/SUIT/coco_annotations/sonosite_val_manual.json

The output mirrors categories/videos/images from sonosite_val_sam2.json so
every image_id stays aligned with the SAM2/UltraSam variants — downstream
test loops and label-quality scripts can swap one annotations file for
another without remapping.

Polygon coordinates are translated from raw video space (CVAT
'original_size', e.g. 960x680 for Sonosite) to post-crop val PNG space
using data/cvat_to_coco.py::crop on each video's first frame, exactly the
same crop the val PNGs were produced with.

CVAT-side fixups:
  - task 61's <source> is mis-typed as CNUH_DC04_BPB1_0051.mp4; rewritten
    to CNUH_DC04_BPB1_0052.mp4.
  - task 65's <source> has a duplicate-upload " (1)" suffix; stripped.
  - source CNUH_DC04_BPB1_0073.mp4 has no entry in sonosite_val_sam2.json
    (excluded from val/test); annotations referencing it are dropped.
  - CVAT polygon `frame` is a project-global counter; subtract the
    cumulative size of preceding tasks to get the local frame index.

Usage:
    PYTHONPATH=. python etc/manual_xml_to_coco.py
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data.cvat_to_coco import crop  # (frame, roi=None) -> (x, y, w, h)


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "data/raw/anno/manual_annotations.xml"
VAL_COCO = ROOT / "data/SUIT/coco_annotations/sonosite_val_sam2.json"
RAW_VID_DIR = ROOT / "data/raw/videos"
OUT_PATH = ROOT / "data/SUIT/coco_annotations/sonosite_val_manual.json"

# Raw label name -> COCO category_id (matches sonosite_val_sam2.json categories).
LABEL_TO_CAT = {
    "C5": 1, "C6": 2, "C7": 3, "C8": 4, "UT": 5,
    "MT": 6, "LT": 7, "SSN": 8, "AD": 9, "PD": 10,
}
TASK_SOURCE_OVERRIDE = {"61": "CNUH_DC04_BPB1_0052.mp4"}
SUFFIX_RE = re.compile(r"\s*\(\d+\)(?=\.[^.]+$)")


def _normalize_source(src: str) -> str:
    return SUFFIX_RE.sub("", src) if src else src


def _crop_bbox(video_name: str,
               cache: dict[str, tuple[int, int, int, int] | None]
               ) -> tuple[int, int, int, int] | None:
    if video_name in cache:
        return cache[video_name]
    raw = RAW_VID_DIR / video_name
    if not raw.exists():
        cache[video_name] = None
        return None
    cap = cv2.VideoCapture(str(raw))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        cache[video_name] = None
        return None
    bbox = crop(frame)
    cache[video_name] = bbox
    return bbox


def _parse_task_meta(root):
    """Return ordered {task_id: {'source','size','offset'}} where offset
    is the cumulative sum of preceding tasks' sizes (CVAT global frame
    counter)."""
    out: dict[str, dict] = {}
    cum = 0
    for t in root.findall(".//meta/project/tasks/task"):
        tid = t.findtext("id")
        src = t.findtext("source") or ""
        if tid in TASK_SOURCE_OVERRIDE:
            src = TASK_SOURCE_OVERRIDE[tid]
        src = _normalize_source(src)
        size = int(t.findtext("size") or 0)
        out[tid] = {"source": src, "size": size, "offset": cum}
        cum += size
    return out


def _track_keyframe(trk):
    """Return the first <polygon keyframe='1' outside='0'> in a track, or
    None. (340 of 340 tracks have at most one such polygon — confirmed.)"""
    polys = sorted(
        (p for p in trk.findall("polygon")
         if p.get("keyframe") == "1" and p.get("outside") == "0"),
        key=lambda p: int(p.get("frame")),
    )
    return polys[0] if polys else None


def main():
    val = json.loads(VAL_COCO.read_text())

    vid2name = {v["id"]: v["name"] for v in val["videos"]}
    img_by_id = {im["id"]: im for im in val["images"]}

    # (video_name, frame_id) -> image_id, so we can attach manual
    # annotations to the val image_ids that already exist.
    name_frame_to_img = {
        (vid2name[im["video_id"]], im["frame_id"]): im["id"]
        for im in val["images"]
    }

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    task_meta = _parse_task_meta(root)

    crop_cache: dict[str, tuple[int, int, int, int] | None] = {}
    annotations = []
    skipped_no_video = 0
    skipped_oob = 0
    skipped_unknown_label = 0
    ann_id = 1

    for trk in root.findall("./track"):
        tid = trk.get("task_id")
        label = trk.get("label")
        if label not in LABEL_TO_CAT:
            skipped_unknown_label += 1
            continue
        meta = task_meta.get(tid)
        if not meta:
            continue
        kp = _track_keyframe(trk)
        if kp is None:
            continue

        bbox = _crop_bbox(meta["source"], crop_cache)
        if bbox is None:
            skipped_no_video += 1
            continue
        cx, cy, cw, ch = bbox

        global_frame = int(kp.get("frame"))
        local_frame = global_frame - meta["offset"]
        key = (meta["source"], local_frame)
        img_id = name_frame_to_img.get(key)
        if img_id is None:
            skipped_oob += 1
            continue

        pts = np.array([
            [float(x), float(y)]
            for x, y in (s.split(",") for s in kp.get("points").split(";"))
        ], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0] - cx, 0.0, float(cw))
        pts[:, 1] = np.clip(pts[:, 1] - cy, 0.0, float(ch))

        if len(pts) < 3:
            continue

        x_min, y_min = pts.min(axis=0).tolist()
        x_max, y_max = pts.max(axis=0).tolist()
        bb_w = float(x_max - x_min)
        bb_h = float(y_max - y_min)
        seg = pts.flatten().tolist()

        annotations.append({
            "id": ann_id,
            "image_id": int(img_id),
            "video_id": int(img_by_id[img_id]["video_id"]),
            "category_id": LABEL_TO_CAT[label],
            "segmentation": [seg],
            "bbox": [float(x_min), float(y_min), bb_w, bb_h],
            "area": float(bb_w * bb_h),
            "iscrowd": 0,
            "instance_id": 1,
        })
        ann_id += 1

    out_doc = {
        "categories": val["categories"],
        "videos": val["videos"],
        "images": val["images"],
        "annotations": annotations,
    }
    OUT_PATH.write_text(json.dumps(out_doc))

    # quick coverage report
    by_video = defaultdict(set)
    for ann in annotations:
        vname = vid2name[ann["video_id"]]
        by_video[vname].add(ann["image_id"])
    print(f"Wrote {len(annotations)} annotations -> {OUT_PATH}")
    print(f"  skipped (no raw video):       {skipped_no_video}")
    print(f"  skipped (frame past val end): {skipped_oob}")
    print(f"  skipped (unknown label):      {skipped_unknown_label}")
    print(f"\nPer-video coverage (frames with at least one manual ann):")
    for v in sorted(vid2name.values()):
        print(f"  {v:34s} {len(by_video.get(v, ())):>4d}")


if __name__ == "__main__":
    main()
