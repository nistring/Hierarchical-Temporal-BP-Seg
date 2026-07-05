import json
import sys
import yaml
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp
from torchvision import tv_tensors
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

sys.path.append("./")
import src.temp_module as rnns
from src.data_loader import UltrasoundTrainDataset, DistributedVideoSampler
from src.utils import post_processing, process_video_stream
from src.losses import ContrastiveLoss, TemporalConsistencyLoss, ExclusionLoss

class TemporalSegmentationModel(nn.Module):
    def __init__(self, encoder_name, segmentation_model_name, num_classes,
                 temporal_model="ConvLSTM", num_layers=1, encoder_depth=5, temporal_depth=1,
                 freeze_encoder=False, kernel_size=(3, 3), dilation=2, conv_type="standard",
                 encoder_weights="imagenet", temporal_upsampling="bilinear",
                 use_hierarchical_fusion: bool = True, **model_kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.temporal_upsampling = temporal_upsampling
        self.use_hierarchical_fusion = use_hierarchical_fusion

        # Initialize segmentation model
        model_args = {
            "encoder_name": encoder_name, "encoder_weights": encoder_weights,
            "classes": num_classes + 1, "in_channels": 1, "encoder_depth": encoder_depth
        }
        model_args.update(model_kwargs)
        model = getattr(smp, segmentation_model_name)(**model_args)

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.head = model.segmentation_head

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.temporal_modules = nn.ModuleList()
        # Start from the bottom level (deepest features) instead of base_depth
        out_channels = self.encoder.out_channels[len(self.encoder.out_channels)-temporal_depth:]
        for i in range(len(out_channels)):
            kwargs = {
                "input_dim": (
                    out_channels[i] + out_channels[i + 1]
                    if (self.use_hierarchical_fusion and i + 1 < len(out_channels))
                    else out_channels[i]
                ),
                "hidden_dim": out_channels[i], "kernel_size": kernel_size,
                "num_layers": num_layers[i] if isinstance(num_layers, list) else num_layers,
                "dilation": dilation, "batch_first": True,
                "conv_type": conv_type,
            }
            temporal_class = getattr(rnns, temporal_model, rnns.ConvLSTM)
            self.temporal_modules.append(temporal_class(**kwargs))

    def forward(self, x, hidden_state=None):
        hidden_state = list(hidden_state) if hidden_state else None
        batch_size, seq_len, c, h, w = x.size()
        x = x.reshape(batch_size * seq_len, c, h, w)

        features = [f.reshape(batch_size, seq_len, *f.shape[1:]) for f in self.encoder(x)]
        temporal_features = features.copy()

        if self.temporal_modules:
            hidden_state = hidden_state or [None] * len(self.temporal_modules)
            start_idx = len(self.encoder.out_channels) - len(self.temporal_modules)

            # Process from deepest to shallowest
            for i in range(len(self.temporal_modules) - 1, -1, -1):
                feature_idx = start_idx + i
                current_features = features[feature_idx]

                # If not the deepest layer, fuse with upsampled features from the layer below
                if self.use_hierarchical_fusion and i < len(self.temporal_modules) - 1:
                    prev_temp_features = temporal_features[feature_idx + 1]

                    upsampled_features = F.interpolate(prev_temp_features.reshape(-1, *prev_temp_features.shape[2:]),
                                                       size=current_features.shape[-2:], mode='bilinear', align_corners=False)
                    upsampled_features = upsampled_features.reshape(batch_size, seq_len, *upsampled_features.shape[1:])

                    # Concatenate along the channel dimension
                    current_features = torch.cat([current_features, upsampled_features], dim=2)

                feature, hidden_state[i] = self.temporal_modules[i](current_features, hidden_state[i])
                temporal_features[feature_idx] = feature

        out = [f.reshape(batch_size * seq_len, *f.shape[2:]) for f in temporal_features]
        out = self.head(self.decoder(*out))
        out = out.reshape(batch_size, seq_len, *out.shape[1:])

        return out, hidden_state

    # ''' For exporting single-frame inference '''
    # def forward(self, x, *hidden_state):
    #     # No temporal dimension during inference
    #     hidden_state = list(hidden_state)
    #     features = [f for f in self.encoder(x / 255.0)]
    #     temporal_features = features.copy()

    #     if self.temporal_modules:
    #         start_idx = len(self.encoder.out_channels) - len(self.temporal_modules)

    #         # Process from deepest to shallowest
    #         for i in range(len(self.temporal_modules) - 1, -1, -1):
    #             feature_idx = start_idx + i
    #             current_features = features[feature_idx]

    #             # If not the deepest layer, fuse with upsampled features from the layer below
    #             if self.use_hierarchical_fusion and i < len(self.temporal_modules) - 1:
    #                 prev_temp_features = temporal_features[feature_idx + 1]
    #                 upsampled_features = F.interpolate(prev_temp_features, size=current_features.shape[-2:], mode='bilinear', align_corners=False)
    #                 # Concatenate along the channel dimension
    #                 current_features = torch.cat([current_features, upsampled_features], dim=1)

    #             feature, hidden_state[i] = self.temporal_modules[i].inference(current_features, hidden_state[i])
    #             temporal_features[feature_idx] = feature

    #     out = self.head(self.decoder(*temporal_features))
    #     out = post_processing(out)[0]
    #     x = process_video_stream(x[0], out)

    #     return [x] + hidden_state


class SegmentationTrainer(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size, learning_rate,
                 num_workers, sequence_length, image_size, truncated_bptt_steps, logdir=None,
                 ce_weight=0.5, temporal_depth=1, temporal_loss_weight=1,
                 negative_weight=100, positive_weight=10, exclusion_weight=0.05,
                 exclusion_groups=None, ckpt_path=None, ce_class_weights=None,
                 manual_annotations_path=None, **kwargs):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.truncated_bptt_steps = truncated_bptt_steps
        self.logdir = logdir
        self.ce_weight = ce_weight
        self.temporal_depth = temporal_depth
        self.temporal_loss_weight = temporal_loss_weight
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.exclusion_weight = exclusion_weight
        self.ckpt_path = ckpt_path
        self.exclusion_groups = exclusion_groups
        # Optional manual-label evaluation: path to a sparse-polygon COCO JSON
        # (e.g. data/SUIT/coco_annotations/sonosite_val_manual.json). When set,
        # `test_step` additionally scores predictions against the manual masks
        # using a partial-label rule — only classes that have a manual label
        # on a given frame contribute to that frame's metric.
        self.manual_annotations_path = manual_annotations_path

        # Data augmentation
        self.transform = v2.Compose([
            v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.2, 0.1), scale=(0.85, 1.15), shear=0.15)]),
            v2.RandomApply([v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(3/4, 4/3))]),
            v2.RandomApply([v2.GaussianNoise()]),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2))]),
            v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.25)]),
            v2.RandomApply([v2.ElasticTransform()]),
        ])

        # Loss functions
        self.loss_tversky = smp.losses.TverskyLoss(mode="multiclass", from_logits=True, alpha=0.5, beta=0.5)
        self.loss_crossentropy = nn.CrossEntropyLoss(
            weight=torch.tensor(ce_class_weights, dtype=torch.float32) if ce_class_weights else None
        )
        self.loss_temporal = TemporalConsistencyLoss() #PerceptualConsistencyLoss()
        self.loss_sparsity = ContrastiveLoss()
        self.loss_exclusion = ExclusionLoss(groups=self.exclusion_groups)

    def show_batch(self, win_size=(80, 100), save=True):
        def _to_vis(imgs, masks):
            imgs = imgs.reshape(-1, *imgs.shape[2:]).cpu()
            masks = masks.reshape(-1, 1, *masks.shape[2:]).bool().float().cpu()
            vis_images = []
            for img, mask in zip(imgs, masks):
                img, mask = to_pil_image(img), to_pil_image(mask)
                img.paste(mask, (0, 0), mask)
                vis_images.append(img)
            return vis_images

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                    shuffle=True, num_workers=self.num_workers, pin_memory=True)
        imgs, targets = next(iter(train_dataloader))
        masks = targets["masks"]
        imgs_aug, masks_aug = imgs.clone(), masks.clone()

        preprocess = v2.Compose([v2.ToImage(), v2.Resize(self.image_size), v2.ToDtype(torch.float32, scale=True)])
        
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            imgs[i], masks[i] = preprocess(img, tv_tensors.Mask(mask))
            imgs_aug[i], masks_aug[i] = self.transform(imgs[i], masks[i])

        def _plot_images(images, filename):
            plt.figure(figsize=win_size)
            grid = torchvision.utils.make_grid(list(map(pil_to_tensor, images)), nrow=self.sequence_length)
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.savefig(filename, bbox_inches="tight", pad_inches=0) if save else plt.show()

        _plot_images(_to_vis(imgs, masks), "figure/original.png")
        _plot_images(_to_vis(imgs_aug, masks_aug), "figure/augmented.png")

    def on_after_batch_transfer(self, batch, batch_idx):
        images, targets = batch
        if self.trainer.training:
            images, targets["masks"] = self.transform(images, tv_tensors.Mask(targets["masks"]))
            targets["masks"] = torch.Tensor(targets["masks"])
        return images, targets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_dataset.batch_size,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2,
                         sampler=DistributedVideoSampler(self.train_dataset, shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_dataset.batch_size,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2,
                         sampler=DistributedVideoSampler(self.val_dataset, shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.ckpt_path:
            return optimizer
            
        warmup_steps = int(len(self.train_dataset) / self.batch_size / self.trainer.num_devices * 3)
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def compute_loss(self, batch, batch_idx, stage):
        images, targets = batch
        
        # Get predictions and temporal feature
        masks, self.hidden_state = self.model(images, self.hidden_state)

        masks_flatten = masks.reshape(-1, *masks.shape[2:])
        targets = targets["masks"].reshape(-1, *targets["masks"].shape[2:]).long()
        
        positive, negative = self.loss_sparsity(masks_flatten)
        losses = {
            'tversky': self.loss_tversky(masks_flatten, targets),
            'crossentropy': self.loss_crossentropy(masks_flatten, targets),
            'temporal': self.loss_temporal(masks),
            'positive': positive,
            'negative': negative,
            'exclusion': self.loss_exclusion(masks_flatten)
        }
        
        total_loss = (losses['tversky'] + self.ce_weight * losses['crossentropy'] +
                     self.temporal_loss_weight * losses['temporal'] +
                     self.positive_weight * positive + self.negative_weight * negative +
                     self.exclusion_weight * losses['exclusion'])

        # Deep-supervision aux loss for EGE-UNet (binary foreground BCE at
        # five scales). The 1-channel `gt_pre` heads were designed for binary
        # segmentation in the original paper; we keep them binary here so the
        # mask-guided GAB module operates as designed, and supervise against
        # the foreground/background mask derived from the multi-class target.
        aux = getattr(self.model, "aux_sigmoids", None)
        if aux is not None and len(aux) > 0:
            # The aux heads are post-sigmoid; recover logits via torch.logit
            # and use BCEWithLogits, which is bf16-autocast-safe (plain BCE
            # on sigmoided inputs is not).
            fg = (targets > 0).float().unsqueeze(1)
            aux_bce = sum(
                F.binary_cross_entropy_with_logits(torch.logit(a, eps=1e-6), fg)
                for a in aux
            ) / len(aux)
            losses['aux_ds'] = aux_bce
            total_loss = total_loss + aux_bce

        # Log all losses
        for name, loss in losses.items():
            self.log(f"{stage}_loss_{name}", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % self.truncated_bptt_steps == 0:
            self.hidden_state = None
        elif self.temporal_depth > 0 and self.hidden_state:
            def detach_nested(item):
                if isinstance(item, list):
                    return [detach_nested(sub_item) for sub_item in item]
                elif isinstance(item, tuple):
                    return tuple(detach_nested(sub_item) for sub_item in item)
                elif isinstance(item, torch.Tensor):
                    return item.detach()
                else:
                    return item
            
            self.hidden_state = detach_nested(self.hidden_state)
        
        # No TBPTT
        # self.hidden_state = None

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self.train_dataset._create_img_list()

    def on_validation_epoch_start(self):
        self.hidden_state = None

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "val")

    def on_test_start(self):
        self.metrics = {metric: [] for metric in ['tp', 'fp', 'fn', 'tn', "video_id"]}
        self.temporal_drift = []  # accumulate per-video mean inter-frame change

        self.video_id = None
        # Prefer the annotations file stem (e.g. 'ge_fold0_val_ultrasam') over
        # the data_dir basename: pooled fold setups point data_dir at
        # `data/SUIT/images/` (yielding the unhelpful name 'images'), but the
        # annotations stem encodes vendor + fold + family.
        ann_path = Path(getattr(self.test_dataset, "annotations_file", "") or "")
        if ann_path.stem:
            self.test_dataset_name = ann_path.stem
        else:
            self.test_dataset_name = Path(self.test_dataset.data_dir).name

        # Manual-label evaluator (partial-label mode). When
        # `manual_annotations_path` is provided we pre-build a per-frame map
        # `image_id -> {coco_category_id: binary_mask}` resized to the model's
        # `image_size`, plus the COCO-cat -> 0-indexed-model-class mapping.
        # During `test_step` we score only those (frame, class) pairs that
        # have a manual label, avoiding penalizing predictions for classes the
        # annotator left unlabeled.
        self._manual_per_frame = {}
        self._manual_cat_to_model = {}
        self.manual_metrics_per_entry = []
        if self.manual_annotations_path:
            from pycocotools.coco import COCO  # noqa: WPS433
            from src.data_loader import category_match
            coco = COCO(self.manual_annotations_path)
            target_h, target_w = self.image_size
            for img_id in coco.imgs:
                ann_ids = coco.getAnnIds(imgIds=img_id)
                if not ann_ids:
                    continue
                anns = coco.loadAnns(ann_ids)
                per_class: dict[int, np.ndarray] = {}
                for ann in anns:
                    cat = ann["category_id"]
                    m = coco.annToMask(ann)
                    m = cv2.resize(m.astype(np.uint8), (target_w, target_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                    per_class[cat] = (per_class[cat] | m) if cat in per_class else m
                if per_class:
                    self._manual_per_frame[img_id] = per_class
            # COCO category_id (1-10) -> model class index (0-7). category_match
            # returns the 1-indexed model class; subtract 1 here once.
            self._manual_cat_to_model = {
                c: category_match[c] - 1 for c in category_match
            }

    def test_step(self, batch):
        images, targets = batch
        if self.video_id != targets["video_id"]:
            self.video_id = targets["video_id"]
            self.hidden_state = None
            for metric in ['tp', 'fp', 'fn', 'tn']:
                self.metrics[metric].append(torch.zeros(self.model.num_classes, device=self.device))
            self.metrics["video_id"].append(self.video_id)
            self.temporal_drift.append([])

        targets["masks"] = F.one_hot(targets["masks"][0].long(),
                                   num_classes=self.model.num_classes+1)[..., 1:].permute(0, 3, 1, 2)

        def _process_masks_and_get_stats(masks, target_masks):
            masks = post_processing(masks)[:, 1:]
            masks_argmax = torch.argmax(masks, dim=1, keepdim=True)
            masks_sum = torch.sum(masks, dim=1, keepdim=True)
            masks = torch.zeros_like(masks)
            masks.scatter_(1, masks_argmax, masks_sum)
            return smp.metrics.get_stats(masks, target_masks, mode="multilabel", threshold=0.5)

        masks, self.hidden_state = self.model(images, self.hidden_state)
        tp, fp, fn, tn = _process_masks_and_get_stats(masks[0], targets["masks"])
        # temporal consistency metric (lower is better): mean L1 difference across time on softmax
        probs = torch.softmax(masks, dim=2)  # (B, T, C, H, W)
        if probs.shape[1] > 1:
            drift = (probs[:, 1:] - probs[:, :-1]).abs().mean()
            self.temporal_drift[-1].append(drift.detach())

        self.metrics['tp'][-1] += tp.sum(0)
        self.metrics['fp'][-1] += fp.sum(0)
        self.metrics['fn'][-1] += fn.sum(0)
        self.metrics['tn'][-1] += tn.sum(0)

        # Partial-label manual eval: score only the (frame, class) pairs that
        # have a manual annotation. We reuse the same argmax-then-thresholded
        # prediction as `_process_masks_and_get_stats` so the manual metric is
        # comparable to the auto-label metric.
        if self._manual_per_frame:
            image_ids = targets.get("image_ids")
            if image_ids is not None:
                if isinstance(image_ids, torch.Tensor):
                    image_ids_list = image_ids.flatten().tolist()
                else:
                    # default_collate may keep a list-of-lists for batch_size=1
                    flat = image_ids[0] if (
                        len(image_ids) == 1 and isinstance(image_ids[0], (list, tuple, torch.Tensor))
                    ) else image_ids
                    image_ids_list = [int(x) for x in flat]
            else:
                image_ids_list = []
            if any(int(i) in self._manual_per_frame for i in image_ids_list):
                m_probs = post_processing(masks[0])[:, 1:]  # (T, C, H, W)
                argmax = torch.argmax(m_probs, dim=1, keepdim=True)
                summed = torch.sum(m_probs, dim=1, keepdim=True)
                preds = torch.zeros_like(m_probs)
                preds.scatter_(1, argmax, summed)
                preds_bin = (preds > 0.5)  # (T, C, H, W)
                for t, img_id in enumerate(image_ids_list):
                    img_id = int(img_id)
                    per_class = self._manual_per_frame.get(img_id)
                    if not per_class:
                        continue
                    for cat_id, manual_np in per_class.items():
                        model_class = self._manual_cat_to_model.get(cat_id)
                        if model_class is None:
                            continue
                        p = preds_bin[t, model_class]
                        m_t = torch.as_tensor(manual_np, device=p.device)
                        tp_pix = int((p & m_t).sum().item())
                        fp_pix = int((p & ~m_t).sum().item())
                        fn_pix = int((~p & m_t).sum().item())
                        denom_iou = max(tp_pix + fp_pix + fn_pix, 1)
                        denom_f1 = max(2 * tp_pix + fp_pix + fn_pix, 1)
                        denom_p = max(tp_pix + fp_pix, 1)
                        denom_s = max(tp_pix + fn_pix, 1)
                        self.manual_metrics_per_entry.append({
                            "image_id": img_id,
                            "category_id": int(cat_id),
                            "model_class": int(model_class),
                            "iou": tp_pix / denom_iou,
                            "f1": (2 * tp_pix) / denom_f1,
                            "precision": tp_pix / denom_p,
                            "sensitivity": tp_pix / denom_s,
                            "tp": tp_pix, "fp": fp_pix, "fn": fn_pix,
                        })

    def on_test_epoch_end(self):
        # Stack metrics
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.metrics[metric] = torch.stack(self.metrics[metric], dim=0)

        # Calculate metrics
        metric_functions = {
            'iou_score': smp.metrics.iou_score,
            'precision': smp.metrics.precision,
            'sensitivity': smp.metrics.sensitivity,
            'f1_score': smp.metrics.f1_score
        }
        
        results = {}
        for name, func in metric_functions.items():
            values = func(self.metrics['tp'], self.metrics['fp'], self.metrics['fn'], self.metrics['tn']).mean(dim=0)
            results[name] = values.cpu().tolist()
            results[f'mean_{name}'] = values.mean().item()

        # Create video-specific results
        results["confusion_matrix"] = {
            str(video_id): {
            "tp": self.metrics["tp"][i].cpu().tolist(),
            "fp": self.metrics["fp"][i].cpu().tolist(), 
            "fn": self.metrics["fn"][i].cpu().tolist(),
            "tn": self.metrics["tn"][i].cpu().tolist()
            }
            for i, video_id in enumerate(self.metrics["video_id"])
        }

        # Temporal consistency report (lower is better)
        if self.temporal_drift:
            drift_means = [torch.stack(d).mean().item() if len(d) else 0.0 for d in self.temporal_drift]
            results["temporal_consistency_mean"] = sum(drift_means) / max(len(drift_means), 1)
            results["temporal_consistency_per_video"] = {
                str(video_id): drift_means[i] for i, video_id in enumerate(self.metrics["video_id"])
            }

        # Manual-label metrics: average per-(frame, class) IoU/F1/Prec/Sens
        # over every manual annotation that the test split actually visits.
        # Frames whose image_id is in the manual COCO but never appear in this
        # test_dataset are silently skipped (they wouldn't have been scored).
        if self.manual_metrics_per_entry:
            entries = self.manual_metrics_per_entry
            n = len(entries)
            results["manual_n_entries"] = n
            for k in ("iou", "f1", "precision", "sensitivity"):
                results[f"manual_mean_{k}"] = sum(e[k] for e in entries) / n
            per_class_groups: dict[int, list[dict]] = {}
            for e in entries:
                per_class_groups.setdefault(e["category_id"], []).append(e)
            results["manual_per_category"] = {
                str(cid): {
                    "n": len(es),
                    "mean_iou": sum(x["iou"] for x in es) / len(es),
                    "mean_f1": sum(x["f1"] for x in es) / len(es),
                    "mean_precision": sum(x["precision"] for x in es) / len(es),
                    "mean_sensitivity": sum(x["sensitivity"] for x in es) / len(es),
                }
                for cid, es in per_class_groups.items()
            }

        # Save results
        logdir = Path(self.logdir)
        results_filename = f"test_results_{self.test_dataset_name}.json"
        with open(logdir / results_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Test results saved to {logdir / results_filename}")

if __name__ == "__main__":
    from pathlib import Path

    # Sample configuration
    with open("configs/sepGRU_batch4.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize datasets
    train_dataset = UltrasoundTrainDataset(
        Path(config["data"]["train"]["data_path"]),
        Path(config["data"]["train"]["annotations_path"]),
        sequence_length=config["data"]["train"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        batch_size=config["data"]["train"]["batch_size"],
        truncated_bptt_steps=config["data"]["train"]["truncated_bptt_steps"],
    )

    # Initialize the model
    model = TemporalSegmentationModel(
        encoder_name=config["model"]["encoder_name"],
        segmentation_model_name=config["model"]["segmentation_model_name"],
        num_classes=config["model"]["num_classes"],
        temporal_model=config["model"]["temporal_model"],
        encoder_depth=config["model"]["encoder_depth"],
        temporal_depth=config["model"]["temporal_depth"],
        freeze_encoder=config["model"].get("freeze_encoder", False),
    )

    # Initialize the trainer
    trainer = SegmentationTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        test_dataset=None,
        batch_size=config["data"]["train"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=config["data"]["train"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        truncated_bptt_steps=config["data"]["train"]["truncated_bptt_steps"],
        logdir=None,
        ce_weight=config["model"]["ce_weight"],
        temporal_depth=config["model"]["temporal_depth"],
        temporal_loss_weight=config["model"].get("temporal_loss_weight", 0.3),
        negative_weight=config["model"].get("negative_weight", 100),
        positive_weight=config["model"].get("positive_weight", 10),
        exclusion_weight=config["model"].get("exclusion_weight", 0.05),
    )

    # Show batch
    trainer.show_batch()
