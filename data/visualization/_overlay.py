"""Shared annotation-overlay helpers for the etc/ visualization scripts.

The 8 model classes (post-collapse) and their colors mirror src/utils.py so
that figures from etc/ scripts match in-model visualizations. CATEGORY_MATCH
collapses the 10 raw COCO category_ids into the 8 model classes, identical
to src/data_loader.py:13.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


# raw category_id (1..10) -> model class id (1..8)
CATEGORY_MATCH = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 6, 9: 7, 10: 8}

# index 0 is the background sentinel — model class ids 1..8 index into [1:]
CLASS_NAMES = ["BG", "C5", "C6", "C7/MT", "C8/LT", "UT", "SSN", "AD", "PD"]

# RGB palette (matches src/utils.py); BGR variant for OpenCV; float-RGB for
# matplotlib. CLASS_*[i] is the color for model class id (i + 1).
_CLASS_RGB_U8 = [
    (192, 255, 0), (0, 255, 192), (64, 0, 255), (255, 0, 64),
    (96, 255, 96), (0, 255, 0), (255, 128, 0), (255, 0, 255),
]
CLASS_BGR = [(b, g, r) for (r, g, b) in _CLASS_RGB_U8]
CLASS_RGB_FLOAT = np.array(_CLASS_RGB_U8, dtype=np.float32) / 255.0


def to_model_class(raw_category_id: int) -> int | None:
    """Map a COCO raw category_id (1..10) to a model class id (1..8)."""
    return CATEGORY_MATCH.get(raw_category_id)


def class_color_bgr(model_class: int) -> tuple[int, int, int]:
    """OpenCV BGR color for a model class id (1..8)."""
    return CLASS_BGR[model_class - 1]


def coco_anns_to_polys(
    anns: Iterable[dict],
    cat_to_model: dict[int, int] = CATEGORY_MATCH,
) -> list[tuple[int, np.ndarray]]:
    """Flatten a list of COCO annotation dicts to ``[(model_class, Nx2 int32), ...]``.

    Skips empty / <3-vertex polygons and unknown raw category_ids. Polygons
    that arrive as flat ``[x1,y1,x2,y2,...]`` lists (COCO standard) are
    reshaped; iscrowd RLE masks are not supported here.
    """
    out: list[tuple[int, np.ndarray]] = []
    for ann in anns:
        cls = cat_to_model.get(ann.get("category_id"))
        if cls is None:
            continue
        seg = ann.get("segmentation") or []
        if isinstance(seg, dict):
            # COCO RLE — not supported in this overlay path
            continue
        for poly in seg:
            if poly is None or len(poly) < 6:
                continue
            pts = (np.asarray(poly, dtype=np.float32)
                   .reshape(-1, 2).round().astype(np.int32))
            out.append((cls, pts))
    return out


def overlay_polygons(
    bgr: np.ndarray,
    polys: list[tuple[int, np.ndarray]],
    alpha: float = 0.45,
    line_width: int = 1,
) -> np.ndarray:
    """Burn semi-transparent fills + outlines for ``polys`` onto ``bgr``.

    polys: ``[(model_class, Nx2 int32 vertices), ...]``. Returns a new
    image; ``bgr`` is not mutated.
    """
    layer = bgr.copy()
    for cls, pts in polys:
        cv2.fillPoly(layer, [pts], class_color_bgr(cls))
    blended = cv2.addWeighted(layer, alpha, bgr, 1.0 - alpha, 0.0)
    for cls, pts in polys:
        cv2.polylines(blended, [pts], True, class_color_bgr(cls),
                      line_width, cv2.LINE_AA)
    return blended


def banner(bgr: np.ndarray, text: str, *,
           org: tuple[int, int] = (8, 18),
           scale: float = 0.45) -> None:
    """Burn a black-stroked white text banner onto ``bgr`` in-place."""
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def classes_summary(polys: list[tuple[int, np.ndarray]]) -> str:
    """Human-readable list of class names present in ``polys``, comma-joined.
    Returns ``"-"`` when empty."""
    classes = sorted({cls for cls, _ in polys})
    if not classes:
        return "-"
    return ", ".join(CLASS_NAMES[c] for c in classes)
