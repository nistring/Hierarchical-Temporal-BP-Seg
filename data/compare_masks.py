"""
Compare SAM2-generated and UltraSam-generated segmentation masks.

Produces side-by-side visualizations for supplementary material showing:
  - Original ultrasound image with bounding box
  - SAM2 mask overlay
  - UltraSam mask overlay

Also computes per-annotation quality metrics:
  - Mask-to-bbox area ratio (flags empty/degenerate masks)
  - Mask area difference between the two models
  - Per-class summary statistics

Usage:
    python data/compare_masks.py \
        --sam2_annotations data/SUIT/coco_annotations/sonosite_train_sam2.json \
        --ultrasam_annotations data/SUIT/coco_annotations/sonosite_train_ultrasam.json \
        --original_annotations data/SUIT/coco_annotations/sonosite_train.json \
        --image_dir data/SUIT/images/sonosite_train \
        --output_dir out/figures/mask_comparison \
        --num_samples 20
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from pycocotools import mask as mask_util


# Class names for brachial plexus nerve elements
CLASS_NAMES = {1: "C5", 2: "C6", 3: "C7/MT", 4: "C8/LT", 5: "UT", 6: "SSN", 7: "AD", 8: "PD"}
# Colors for each class (RGB, 0-1)
CLASS_COLORS = {
    1: (0.75, 1.0, 0.0), 2: (0.0, 1.0, 0.75), 3: (0.25, 0.0, 1.0), 4: (1.0, 0.0, 0.25),
    5: (0.38, 1.0, 0.38), 6: (0.0, 1.0, 0.0), 7: (1.0, 0.5, 0.0), 8: (1.0, 0.0, 1.0),
}


def polygon_to_mask(segmentation, h, w):
    """Convert COCO polygon segmentation to binary mask."""
    if not segmentation or not segmentation[0]:
        return np.zeros((h, w), dtype=np.uint8)
    rle = mask_util.frPyObjects(segmentation, h, w)
    mask = mask_util.decode(mask_util.merge(rle))
    return mask


def compute_bbox_area(bbox):
    """Compute area from COCO bbox [x, y, w, h]."""
    return bbox[2] * bbox[3]


def overlay_mask(ax, img, mask, color, alpha=0.4):
    """Overlay a colored mask on an image axis."""
    overlay = img.copy()
    mask_bool = mask > 0
    for c in range(3):
        overlay[:, :, c] = np.where(mask_bool,
                                     img[:, :, c] * (1 - alpha) + int(color[c] * 255) * alpha,
                                     img[:, :, c])
    # Draw contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, tuple(int(c * 255) for c in color), 2)
    ax.imshow(overlay)


def visualize_comparison(img, bbox, sam2_mask, ultrasam_mask, category_id, img_id, output_path):
    """Create a 3-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    color = CLASS_COLORS.get(category_id, (1, 1, 0))
    class_name = CLASS_NAMES.get(category_id, f"class_{category_id}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Panel 1: Original image with bounding box
    axes[0].imshow(img_rgb)
    x, y, w, h = bbox
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"Input + BBox ({class_name})", fontsize=12)
    axes[0].axis("off")

    # Panel 2: SAM2 mask
    sam2_area = sam2_mask.sum()
    bbox_area = compute_bbox_area(bbox)
    ratio_sam2 = sam2_area / max(bbox_area, 1) * 100
    overlay_mask(axes[1], img_rgb, sam2_mask, color)
    axes[1].set_title(f"SAM2 (area ratio: {ratio_sam2:.0f}%)", fontsize=12)
    axes[1].axis("off")

    # Panel 3: UltraSam mask
    ultra_area = ultrasam_mask.sum()
    ratio_ultra = ultra_area / max(bbox_area, 1) * 100
    overlay_mask(axes[2], img_rgb, ultrasam_mask, color)
    axes[2].set_title(f"UltraSam (area ratio: {ratio_ultra:.0f}%)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(f"Image {img_id} — {class_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_statistics(original_coco, sam2_coco, ultrasam_coco, image_dir):
    """Compute per-annotation quality metrics for both models."""
    stats = {cat_id: {"sam2_empty": 0, "ultra_empty": 0, "sam2_areas": [], "ultra_areas": [],
                       "sam2_ratios": [], "ultra_ratios": [], "total": 0}
             for cat_id in range(1, 11)}

    for img_id in sorted(original_coco.imgs.keys()):
        ann_ids = original_coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            continue

        img_info = original_coco.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]
        orig_anns = original_coco.loadAnns(ann_ids)

        sam2_anns = sam2_coco.loadAnns(ann_ids)
        ultra_anns = ultrasam_coco.loadAnns(ann_ids)

        for orig_ann, sam2_ann, ultra_ann in zip(orig_anns, sam2_anns, ultra_anns):
            cat_id = orig_ann["category_id"]
            bbox_area = compute_bbox_area(orig_ann["bbox"])
            if bbox_area == 0:
                continue

            s = stats[cat_id]
            s["total"] += 1

            # SAM2
            sam2_seg = sam2_ann.get("segmentation", [])
            sam2_mask = polygon_to_mask(sam2_seg, h, w)
            sam2_area = sam2_mask.sum()
            s["sam2_areas"].append(sam2_area)
            s["sam2_ratios"].append(sam2_area / bbox_area)
            if sam2_area < bbox_area * 0.05:
                s["sam2_empty"] += 1

            # UltraSam
            ultra_seg = ultra_ann.get("segmentation", [])
            ultra_mask = polygon_to_mask(ultra_seg, h, w)
            ultra_area = ultra_mask.sum()
            s["ultra_areas"].append(ultra_area)
            s["ultra_ratios"].append(ultra_area / bbox_area)
            if ultra_area < bbox_area * 0.05:
                s["ultra_empty"] += 1

    return stats


def print_statistics(stats):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print(f"{'Class':<10} {'Total':>6} {'SAM2 empty':>12} {'Ultra empty':>12} "
          f"{'SAM2 ratio':>12} {'Ultra ratio':>12}")
    print("-" * 80)

    total_sam2_empty = 0
    total_ultra_empty = 0
    total_count = 0

    for cat_id in range(1, 11):
        s = stats[cat_id]
        if s["total"] == 0:
            continue
        name = CLASS_NAMES.get(cat_id, f"cat_{cat_id}")
        sam2_ratio = np.mean(s["sam2_ratios"]) if s["sam2_ratios"] else 0
        ultra_ratio = np.mean(s["ultra_ratios"]) if s["ultra_ratios"] else 0

        print(f"{name:<10} {s['total']:>6} "
              f"{s['sam2_empty']:>8} ({100*s['sam2_empty']/s['total']:>4.1f}%) "
              f"{s['ultra_empty']:>8} ({100*s['ultra_empty']/s['total']:>4.1f}%) "
              f"{sam2_ratio:>11.2f} {ultra_ratio:>11.2f}")

        total_sam2_empty += s["sam2_empty"]
        total_ultra_empty += s["ultra_empty"]
        total_count += s["total"]

    print("-" * 80)
    print(f"{'TOTAL':<10} {total_count:>6} "
          f"{total_sam2_empty:>8} ({100*total_sam2_empty/max(total_count,1):>4.1f}%) "
          f"{total_ultra_empty:>8} ({100*total_ultra_empty/max(total_count,1):>4.1f}%)")
    print("=" * 80)


def select_comparison_samples(original_coco, sam2_coco, ultrasam_coco, image_dir, num_samples=20):
    """Select informative samples for visual comparison.
    Prioritize: (1) SAM2 failures, (2) large quality differences, (3) diverse classes."""
    candidates = []

    for img_id in sorted(original_coco.imgs.keys()):
        ann_ids = original_coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            continue

        img_info = original_coco.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]
        orig_anns = original_coco.loadAnns(ann_ids)
        sam2_anns = sam2_coco.loadAnns(ann_ids)
        ultra_anns = ultrasam_coco.loadAnns(ann_ids)

        for orig_ann, sam2_ann, ultra_ann in zip(orig_anns, sam2_anns, ultra_anns):
            bbox_area = compute_bbox_area(orig_ann["bbox"])
            if bbox_area == 0:
                continue

            sam2_seg = sam2_ann.get("segmentation", [])
            ultra_seg = ultra_ann.get("segmentation", [])
            sam2_area = polygon_to_mask(sam2_seg, h, w).sum()
            ultra_area = polygon_to_mask(ultra_seg, h, w).sum()

            sam2_ratio = sam2_area / bbox_area
            ultra_ratio = ultra_area / bbox_area
            diff = abs(ultra_ratio - sam2_ratio)

            # Priority: SAM2 failed but UltraSam succeeded
            sam2_failed = sam2_ratio < 0.05
            ultra_ok = ultra_ratio >= 0.05
            priority = 0
            if sam2_failed and ultra_ok:
                priority = 2  # Highest: SAM2 failure fixed by UltraSam
            elif diff > 0.3:
                priority = 1  # Large difference

            candidates.append({
                "img_id": img_id, "ann_id": orig_ann["id"],
                "category_id": orig_ann["category_id"],
                "priority": priority, "diff": diff,
                "sam2_ratio": sam2_ratio, "ultra_ratio": ultra_ratio,
            })

    # Sort by priority (desc), then diff (desc)
    candidates.sort(key=lambda x: (x["priority"], x["diff"]), reverse=True)

    # Ensure class diversity in selected samples
    selected = []
    class_count = {}
    for c in candidates:
        cat = c["category_id"]
        if class_count.get(cat, 0) >= num_samples // 4:
            continue
        selected.append(c)
        class_count[cat] = class_count.get(cat, 0) + 1
        if len(selected) >= num_samples:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Compare SAM2 vs UltraSam masks")
    parser.add_argument("--sam2_annotations", type=str, required=True,
                        help="Path to SAM2-generated annotations (e.g., sonosite_train_sam2.json)")
    parser.add_argument("--ultrasam_annotations", type=str, required=True,
                        help="Path to UltraSam-generated annotations (e.g., sonosite_train_ultrasam.json)")
    parser.add_argument("--original_annotations", type=str, required=True,
                        help="Path to original bbox annotations (e.g., train.json)")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="out/figures/mask_comparison")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of comparison figures to generate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    print("Loading annotations...")
    original_coco = COCO(args.original_annotations)
    sam2_coco = COCO(args.sam2_annotations)
    ultrasam_coco = COCO(args.ultrasam_annotations)

    # Compute and print statistics
    print("Computing statistics...")
    stats = compute_statistics(original_coco, sam2_coco, ultrasam_coco, image_dir)
    print_statistics(stats)

    # Save statistics to JSON
    stats_serializable = {}
    for cat_id, s in stats.items():
        if s["total"] == 0:
            continue
        stats_serializable[CLASS_NAMES.get(cat_id, f"cat_{cat_id}")] = {
            "total": s["total"],
            "sam2_empty": s["sam2_empty"],
            "ultrasam_empty": s["ultra_empty"],
            "sam2_empty_pct": round(100 * s["sam2_empty"] / s["total"], 2),
            "ultrasam_empty_pct": round(100 * s["ultra_empty"] / s["total"], 2),
            "sam2_mean_ratio": round(float(np.mean(s["sam2_ratios"])), 4) if s["sam2_ratios"] else 0,
            "ultrasam_mean_ratio": round(float(np.mean(s["ultra_ratios"])), 4) if s["ultra_ratios"] else 0,
        }
    with open(output_dir / "mask_quality_statistics.json", "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"Statistics saved to {output_dir / 'mask_quality_statistics.json'}")

    # Select and generate comparison figures
    print(f"\nSelecting {args.num_samples} samples for visual comparison...")
    samples = select_comparison_samples(
        original_coco, sam2_coco, ultrasam_coco, image_dir, args.num_samples
    )

    for i, sample in enumerate(samples):
        img_info = original_coco.loadImgs(sample["img_id"])[0]
        h, w = img_info["height"], img_info["width"]
        img_path = image_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        orig_ann = original_coco.loadAnns([sample["ann_id"]])[0]
        sam2_ann = sam2_coco.loadAnns([sample["ann_id"]])[0]
        ultra_ann = ultrasam_coco.loadAnns([sample["ann_id"]])[0]

        sam2_mask = polygon_to_mask(sam2_ann.get("segmentation", []), h, w)
        ultra_mask = polygon_to_mask(ultra_ann.get("segmentation", []), h, w)

        class_name = CLASS_NAMES.get(sample["category_id"], "unknown")
        safe_name = class_name.replace("/", "-")
        output_path = output_dir / f"compare_{i:02d}_img{sample['img_id']}_{safe_name}.png"

        visualize_comparison(
            img, orig_ann["bbox"], sam2_mask, ultra_mask,
            sample["category_id"], sample["img_id"], output_path
        )

    print(f"\n{len(samples)} comparison figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
