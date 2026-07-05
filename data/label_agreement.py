"""R1.2 label-accuracy analysis: agreement of semi-auto labels vs manual GT.

For every manual annotation (319 sparse hand-drawn polygons on the Sonosite val
set), find the matching semi-auto annotation by (image_id, category_id) and
compute mask IoU and Dice. SAM2 and UltraSam share the same upstream bbox seeds,
so both are scored on the identical matched subset — a clean apples-to-apples
mask-quality comparison given the same prompt.

Coverage (fraction of manual anns that have a seed-derived auto label at all) is
reported separately: a manual ann with no matching auto label is a tracker-seed
coverage gap, not a mask-quality failure, so it is excluded from the IoU/Dice
means and counted under coverage instead.

Outputs:
  - out/label_agreement.csv      (per-class IoU/Dice/coverage, both models)
  - stdout markdown table

Usage:
    python data/label_agreement.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

COCO_DIR = Path("data/SUIT/coco_annotations")
MANUAL = COCO_DIR / "sonosite_val_manual.json"
AUTO = {"SAM2": COCO_DIR / "sonosite_val_sam2.json",
        "UltraSam": COCO_DIR / "sonosite_val_ultrasam.json"}
OUT_CSV = Path("out/label_agreement.csv")

# 10 COCO categories in the SUIT annotation space (pre-merge).
CAT_NAMES = {1: "C5", 2: "C6", 3: "C7", 4: "C8", 5: "UT",
             6: "MT", 7: "LT", 8: "SSN", 9: "AD", 10: "PD"}


def poly_to_mask(seg, h, w):
    """COCO polygon -> binary mask (uint8). Empty/degenerate -> zeros."""
    if not seg:
        return np.zeros((h, w), dtype=np.uint8)
    # seg may be a list of polygons; drop any with <3 points (need >=6 coords).
    polys = [p for p in seg if isinstance(p, list) and len(p) >= 6]
    if not polys:
        return np.zeros((h, w), dtype=np.uint8)
    rle = mask_util.frPyObjects(polys, h, w)
    return mask_util.decode(mask_util.merge(rle))


def iou_dice(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = sa + sb - inter
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (sa + sb) if (sa + sb) > 0 else 0.0
    return float(iou), float(dice)


def index_auto(path):
    """(image_id, category_id) -> merged polygon list."""
    d = json.loads(Path(path).read_text())
    by_key = defaultdict(list)
    for a in d["annotations"]:
        seg = a.get("segmentation") or []
        if seg:
            by_key[(a["image_id"], a["category_id"])].extend(seg)
    return by_key


def main():
    manual = json.loads(MANUAL.read_text())
    img_hw = {im["id"]: (im["height"], im["width"]) for im in manual["images"]}
    man_anns = manual["annotations"]
    n_total = defaultdict(int)
    for a in man_anns:
        n_total[a["category_id"]] += 1

    rows = []
    per_model = {}
    for model, path in AUTO.items():
        auto = index_auto(path)
        ious = defaultdict(list)
        dices = defaultdict(list)
        matched = defaultdict(int)
        for a in man_anns:
            cid = a["category_id"]
            key = (a["image_id"], cid)
            if key not in auto:
                continue  # tracker-seed coverage gap; counted via matched<total
            h, w = img_hw[a["image_id"]]
            m_mask = poly_to_mask(a.get("segmentation") or [], h, w)
            a_mask = poly_to_mask(auto[key], h, w)
            iou, dice = iou_dice(m_mask, a_mask)
            ious[cid].append(iou)
            dices[cid].append(dice)
            matched[cid] += 1
        per_model[model] = dict(ious=ious, dices=dices, matched=matched)

    # Build a per-class table across both models.
    print(f"\nLabel-vs-manual agreement on {len(man_anns)} manual annotations "
          f"(matched subset only; coverage shown separately).\n")
    header = f"| {'Class':<6} | {'n man':>5} |"
    for m in AUTO:
        header += f" {m+' IoU':>12} | {m+' Dice':>13} | {m+' cov':>10} |"
    print(header)
    sep = "|" + "-" * 8 + "|" + "-" * 7 + "|"
    for _ in AUTO:
        sep += "-" * 14 + "|" + "-" * 15 + "|" + "-" * 12 + "|"
    print(sep)

    micro = {m: {"iou": [], "dice": []} for m in AUTO}
    macro = {m: {"iou": [], "dice": []} for m in AUTO}
    for cid in range(1, 11):
        ntot = n_total.get(cid, 0)
        if ntot == 0:
            continue
        line = f"| {CAT_NAMES[cid]:<6} | {ntot:>5} |"
        csv_row = {"class": CAT_NAMES[cid], "n_manual": ntot}
        for m in AUTO:
            pm = per_model[m]
            iv = pm["ious"].get(cid, [])
            dv = pm["dices"].get(cid, [])
            mt = pm["matched"].get(cid, 0)
            iou_m = float(np.mean(iv)) if iv else float("nan")
            dice_m = float(np.mean(dv)) if dv else float("nan")
            cov = mt / ntot if ntot else 0.0
            line += f" {iou_m:>12.3f} | {dice_m:>13.3f} | {mt:>3}/{ntot:<3}={cov*100:>3.0f}% |"
            csv_row[f"{m}_iou"] = round(iou_m, 4)
            csv_row[f"{m}_dice"] = round(dice_m, 4)
            csv_row[f"{m}_coverage"] = round(cov, 4)
            csv_row[f"{m}_matched"] = mt
            micro[m]["iou"].extend(iv)
            micro[m]["dice"].extend(dv)
            if iv:
                macro[m]["iou"].append(iou_m)
                macro[m]["dice"].append(dice_m)
        print(line)
        rows.append(csv_row)

    # Overall micro (per-annotation) and macro (per-class) means.
    print(sep)
    n_man = len(man_anns)
    micro_line = f"| {'micro':<6} | {n_man:>5} |"
    macro_line = f"| {'macro':<6} | {'':>5} |"
    overall_rows = {"micro": {"class": "micro_all", "n_manual": n_man},
                    "macro": {"class": "macro_mean", "n_manual": ""}}
    for m in AUTO:
        mi_i = float(np.mean(micro[m]["iou"])) if micro[m]["iou"] else float("nan")
        mi_d = float(np.mean(micro[m]["dice"])) if micro[m]["dice"] else float("nan")
        ma_i = float(np.mean(macro[m]["iou"])) if macro[m]["iou"] else float("nan")
        ma_d = float(np.mean(macro[m]["dice"])) if macro[m]["dice"] else float("nan")
        tot_matched = sum(per_model[m]["matched"].values())
        cov = tot_matched / n_man
        micro_line += f" {mi_i:>12.3f} | {mi_d:>13.3f} | {tot_matched:>3}/{n_man:<3}={cov*100:>3.0f}% |"
        macro_line += f" {ma_i:>12.3f} | {ma_d:>13.3f} | {'':>10} |"
        overall_rows["micro"][f"{m}_iou"] = round(mi_i, 4)
        overall_rows["micro"][f"{m}_dice"] = round(mi_d, 4)
        overall_rows["micro"][f"{m}_coverage"] = round(cov, 4)
        overall_rows["micro"][f"{m}_matched"] = tot_matched
        overall_rows["macro"][f"{m}_iou"] = round(ma_i, 4)
        overall_rows["macro"][f"{m}_dice"] = round(ma_d, 4)
    print(micro_line)
    print(macro_line)
    rows.append(overall_rows["micro"])
    rows.append(overall_rows["macro"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    keys = ["class", "n_manual"]
    for m in AUTO:
        keys += [f"{m}_iou", f"{m}_dice", f"{m}_coverage", f"{m}_matched"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
