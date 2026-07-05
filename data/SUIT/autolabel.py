import argparse
from pathlib import Path
import torch
from ultralytics import SAM
from pycocotools.coco import COCO
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--steps", nargs="+", default=["train", "val"])
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

model = SAM("sam2.1_l.pt")
model.to(args.device)

for step in args.steps:
    annotations_file = Path("coco_annotations") / f"{step}.json"
    coco = COCO(annotations_file)
    for id in tqdm(list(sorted(coco.imgs.keys())), desc=step):
        img = coco.loadImgs(id)[0]
        path = img['file_name']
        ann_ids = coco.getAnnIds(imgIds=id)
        if ann_ids:
            coco_annotation = coco.loadAnns(ann_ids)
            labels = [ann['category_id'] - 1 for ann in coco_annotation]
            bboxes = torch.Tensor([ann['bbox'] for ann in coco_annotation])
            bboxes[:, 2:] += bboxes[:, :2]
            result = model(f"images/{step}/{path}", bboxes=bboxes, verbose=False, save=False, labels=labels, conf=0.0)[0]
            segments = result.masks.xy

            for ann, segment in zip(coco_annotation, segments):
                ann['segmentation'] = [segment.flatten().tolist()]

    out = Path("coco_annotations") / f"{step}_sam2.json"
    with open(out, "w") as f:
        json.dump(coco.dataset, f)
    print(f"[{step}] wrote {out}")