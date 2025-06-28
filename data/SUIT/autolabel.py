from pathlib import Path
import torch
from ultralytics import SAM
from pycocotools.coco import COCO
import json  # Add import for json
from tqdm import tqdm

# Load a model
model = SAM("sam2.1_l.pt")

for step in ["val"]:#"GE_train", "GE_val", "mindray_train", "mindray_val", "val"]:
    annotations_file = Path("coco_annotations") / f"{step}.json"
    coco = COCO(annotations_file)
    for id in tqdm(list(sorted(coco.imgs.keys()))):  # noqa
        img = coco.loadImgs(id)[0]
        path = img['file_name']
        ann_ids = coco.getAnnIds(imgIds=id)
        if ann_ids:
            coco_annotation = coco.loadAnns(ann_ids)
            labels = [ann['category_id'] - 1 for ann in coco_annotation]
            bboxes = torch.Tensor([ann['bbox'] for ann in coco_annotation])
            bboxes[:, 2:] += bboxes[:, :2]
            result = model(f"images/{step}/{path}", bboxes=bboxes, verbose=False, save=False, labels=labels)[0]
            segments = result.masks.xy
            
            # Add predicted segmentation results to coco_annotations
            for ann, segment in zip(coco_annotation, segments):
                ann['segmentation'] = [segment.flatten().tolist()]

    # Save updated annotations to a new JSON file
    with open(Path("coco_annotations") / f"{step}_updated.json", "w") as f:
        json.dump(coco.dataset, f)