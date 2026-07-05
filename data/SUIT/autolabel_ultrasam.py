"""
UltraSam auto-labeling. Run from the UltraSam repo directory with the ultrasam conda env.

Usage:
    cd /home/nistring/UltraSam
    PYTHONPATH=. /home/nistring/.conda/envs/ultrasam/bin/python \
        /home/nistring/Hierarchical-Temporal-BP-Seg/data/SUIT/autolabel_ultrasam.py \
        --steps train val --device cuda:0
"""

import argparse
import json
from pathlib import Path

import sys
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

SUIT_ROOT = Path(__file__).resolve().parent
ULTRASAM_ROOT = Path("/home/nistring/UltraSam")
sys.path.insert(0, str(ULTRASAM_ROOT))

from mmdet.utils import register_all_modules
register_all_modules()
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.structures.bbox import HorizontalBoxes
from mmcv.transforms import Compose

# UltraSam relies on MonkeyPatchHook (configs/_base_/models/sam.py:127) to swap
# F.multi_head_attention_forward for a version that handles SAMAttention's
# downsample-rate trick. Since we build the model directly (no mim runner),
# the hook never fires — apply the patch manually.
import torch.nn.functional as F
from endosam.models.utils.custom_functional import multi_head_attention_forward
F.multi_head_attention_forward = multi_head_attention_forward


def build_model(device):
    cfg = Config.fromfile(str(ULTRASAM_ROOT / "configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py"))
    model = MODELS.build(cfg.model)
    ckpt = torch.load(str(ULTRASAM_ROOT / "UltraSam.pth"), map_location="cpu", weights_only=False)
    # UltraSam.pth is wrapped as {'meta', 'state_dict', ...}. Loading the outer
    # dict silently yields a randomly-initialised model (strict=False swallows
    # the 293 missing keys). Extract state_dict first.
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    msg = model.load_state_dict(state_dict, strict=False)
    assert len(msg.missing_keys) == 0, f"missing {len(msg.missing_keys)} weight keys: {msg.missing_keys[:3]}"
    return model.eval().to(device), cfg


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    return [contour.flatten().tolist()] if contour.shape[0] >= 3 else []


@torch.no_grad()
def autolabel(steps, device):
    model, cfg = build_model(device)
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline_pre = Compose(pipeline[:2])    # LoadImage + Resize
    pipeline_post = Compose(pipeline[3:])   # GetPoint* + GetPrompt + Pack

    for step in steps:
        coco = COCO(str(SUIT_ROOT / f"coco_annotations/{step}.json"))
        empty_count, total_count = 0, 0

        for img_id in tqdm(sorted(coco.imgs.keys()), desc=f"[{step}]"):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            if not ann_ids:
                continue

            img_info = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(ann_ids)
            bboxes = np.array([a["bbox"] for a in anns], dtype=np.float32)
            bboxes[:, 2:] += bboxes[:, :2]  # xywh -> xyxy

            # Load + resize image
            data = pipeline_pre({"img_path": str(SUIT_ROOT / f"images/{step}/{img_info['file_name']}"), "img_id": img_id})

            # Inject bboxes in ORIGINAL coords (bypassing LoadAnnotations which requires existing masks).
            # Downstream GetPointBox uses scale_factor to project to post-resize coords, so we must NOT
            # pre-scale here — doing so would double-scale the prompt and push it outside the target.
            data["gt_bboxes"] = HorizontalBoxes(torch.tensor(bboxes, dtype=torch.float32))
            data["gt_bboxes_labels"] = np.array([a["category_id"] - 1 for a in anns], dtype=np.int64)
            data["gt_masks"] = np.zeros((len(anns), *data["img_shape"]), dtype=np.uint8)

            # Generate prompts + pack
            data = pipeline_post(data)

            # Preprocess + inference
            batch = model.data_preprocessor({"inputs": [data["inputs"]], "data_samples": [data["data_samples"]]})
            results = model.predict(batch["inputs"], batch["data_samples"], rescale=True)

            # Extract masks
            pred = results[0].pred_instances
            ori_h, ori_w = img_info["height"], img_info["width"]

            if hasattr(pred, "masks") and pred.masks.numel() > 0:
                for ann, mask in zip(anns, pred.masks.cpu().numpy()):
                    total_count += 1
                    if mask.shape != (ori_h, ori_w):
                        mask = cv2.resize(mask.astype(np.uint8), (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
                    poly = mask_to_polygon(mask)
                    ann["segmentation"] = poly
                    if not poly:
                        empty_count += 1
            else:
                for ann in anns:
                    total_count += 1
                    empty_count += 1
                    ann["segmentation"] = []

        out = SUIT_ROOT / f"coco_annotations/{step}_ultrasam.json"
        with open(out, "w") as f:
            json.dump(coco.dataset, f)
        print(f"[{step}] Saved {out} | {total_count} anns, {empty_count} empty ({100*empty_count/max(total_count,1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=["train", "val"])
    parser.add_argument("--device", default="cuda:0")
    autolabel(**vars(parser.parse_args()))
