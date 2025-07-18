import json
import sys
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
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
import src.rnns as rnns
from src.data_loader import UltrasoundTrainDataset, DistributedVideoSampler
from src.utils import post_processing
from src.losses import TemporalConsistencyLoss, SparsityLoss

class TemporalSegmentationModel(nn.Module):
    def __init__(self, encoder_name, segmentation_model_name, num_classes, image_size,
                 temporal_model="ConvLSTM", num_layers=1, encoder_depth=5, temporal_depth=1,
                 freeze_encoder=False, **model_kwargs):
        super().__init__()
        assert encoder_depth > temporal_depth, "Encoder depth should be greater than temporal depth."
        
        self.num_classes = num_classes
        
        # Initialize segmentation model
        model_args = {
            "encoder_name": encoder_name, "encoder_weights": "imagenet",
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

        # Initialize temporal models
        self.temporal_models = nn.ModuleList()
        h, w = image_size
        base_depth = 2 if self.encoder.out_channels[1] == 0 else 1
        
        for i, out_channel in enumerate(self.encoder.out_channels[base_depth:base_depth + temporal_depth]):
            kwargs = {
                "input_size": (h // (2 ** (i + base_depth)), w // (2 ** (i + base_depth))),
                "input_dim": out_channel, "hidden_dim": out_channel, "kernel_size": (3, 3),
                "num_layers": num_layers, "dilation": 2, "batch_first": True,
            }
            temporal_class = getattr(rnns, temporal_model, rnns.ConvLSTM)
            self.temporal_models.append(temporal_class(**kwargs))

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, c, h, w = x.size()
        x = x.reshape(batch_size * seq_len, c, h, w)

        features = [f.reshape(batch_size, seq_len, *f.shape[1:]) for f in self.encoder(x)]

        if not self.temporal_models:
            temporal_outs = features
        else:
            temporal_outs = features[:2]
            hidden_state = hidden_state or [None] * len(self.temporal_models)
            
            for i, temporal_model in enumerate(self.temporal_models):
                temporal_out, hidden_state[i] = temporal_model(features[2 + i], hidden_state[i])
                temporal_outs.append(temporal_out)
            temporal_outs.extend(features[2 + len(self.temporal_models):])

        temporal_outs = [f.reshape(batch_size * seq_len, *f.shape[2:]) for f in temporal_outs]
        x = self.head(self.decoder(temporal_outs))
        return x.reshape(batch_size, seq_len, *x.shape[1:]), hidden_state


class SegmentationTrainer(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size, learning_rate,
                 num_workers, sequence_length, image_size, truncated_bptt_steps, logdir=None,
                 alpha=0.5, encoder_depth=5, temporal_depth=1, temporal_loss_weight=1,
                 sparsity_weight=0.05, ckpt_path=None, **kwargs):
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
        self.alpha = alpha
        self.temporal_depth = temporal_depth
        self.temporal_loss_weight = temporal_loss_weight
        self.sparsity_weight = sparsity_weight
        self.ckpt_path = ckpt_path

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
        self.loss_crossentropy = nn.CrossEntropyLoss()
        self.loss_temporal = TemporalConsistencyLoss()
        self.loss_sparsity = SparsityLoss()

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

        _plot_images(_to_vis(imgs, masks), "original.png")
        _plot_images(_to_vis(imgs_aug, masks_aug), "augmented.png")

    def on_after_batch_transfer(self, batch, batch_idx):
        images, targets = batch
        if self.trainer.training:
            images, targets["masks"] = self.transform(images, tv_tensors.Mask(targets["masks"]))
            targets["masks"] = torch.Tensor(targets["masks"])
        return images, targets

    def _create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                         num_workers=self.num_workers, pin_memory=True,
                         sampler=DistributedVideoSampler(dataset, batch_size=self.batch_size))

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.ckpt_path:
            return optimizer
            
        warmup_steps = int(len(self.train_dataset) / self.batch_size / self.trainer.num_devices / 
                          self.trainer.accumulate_grad_batches * 5)
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def compute_loss(self, batch, batch_idx, stage):
        images, targets = batch
        masks, self.hidden_state = self.model(images, self.hidden_state)

        masks_original = masks
        masks = masks.reshape(-1, *masks.shape[2:])
        targets = targets["masks"].reshape(-1, *targets["masks"].shape[2:]).long()
        
        losses = {
            'tversky': self.loss_tversky(masks, targets),
            'crossentropy': self.loss_crossentropy(masks, targets),
            'temporal': self.loss_temporal(masks_original),
            'sparsity': self.loss_sparsity(masks)
        }
        
        total_loss = (losses['tversky'] + self.alpha * losses['crossentropy'] + 
                     self.temporal_loss_weight * losses['temporal'] + 
                     self.sparsity_weight * losses['sparsity'])

        # Log all losses
        for name, loss in losses.items():
            self.log(f"{stage}_loss_{name}", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % self.truncated_bptt_steps == 0:
            self.hidden_state = None
        elif self.temporal_depth > 0 and self.hidden_state:
            self.hidden_state = [[h.detach() for h in hs] for hs in self.hidden_state]

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "train")

    def on_train_epoch_end(self):
        self.train_dataset._create_img_list()

    def on_validation_epoch_start(self):
        self.hidden_state = None

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "val")

    def on_test_start(self):
        n = self.truncated_bptt_steps
        self.metrics = {metric: [[[] for _ in range(n)] for _ in range(n)] 
                       for metric in ['tp', 'fp', 'fn', 'tn']}
        self.video_id = None
        self.test_dataset_name = Path(self.test_dataset.data_dir).name

    def test_step(self, batch):
        images, targets = batch
        if self.video_id != targets["video_id"]:
            self.video_id = targets["video_id"]
            self.hidden_states = []

        if len(self.hidden_states) < self.truncated_bptt_steps:
            self.hidden_states.append(None)

        targets["masks"] = F.one_hot(targets["masks"][0].long(), 
                                   num_classes=self.model.num_classes+1)[..., 1:].permute(0, 3, 1, 2)
        
        def _process_masks_and_get_stats(masks, target_masks):
            masks = post_processing(masks)[:, 1:]
            masks_argmax = torch.argmax(masks, dim=1, keepdim=True)
            masks_sum = torch.sum(masks, dim=1, keepdim=True)
            masks = torch.zeros_like(masks)
            masks.scatter_(1, masks_argmax, masks_sum)
            return smp.metrics.get_stats(masks, target_masks, mode="multilabel", threshold=0.5)
        
        for i, hidden_state in enumerate(self.hidden_states):
            masks, self.hidden_states[i] = self.model(images, hidden_state)
            tp, fp, fn, tn = _process_masks_and_get_stats(masks, targets["masks"])

            for j in range(i, len(self.hidden_states)):
                self.metrics['tp'][i][j].append(tp)
                self.metrics['fp'][i][j].append(fp)
                self.metrics['fn'][i][j].append(fn)
                self.metrics['tn'][i][j].append(tn)

    def on_test_epoch_end(self):
        n = self.truncated_bptt_steps
        results = {}
        
        for metric_name in ['iou_score', 'f1_score', 'precision', 'sensitivity']:
            scores = torch.zeros(n, n, self.model.num_classes)
            mean_scores = torch.zeros(n, n)
            
            for i in range(n):
                for j in range(i, n):
                    # Concatenate metrics
                    tp = torch.cat(self.metrics['tp'][i][j])
                    fp = torch.cat(self.metrics['fp'][i][j])
                    fn = torch.cat(self.metrics['fn'][i][j])
                    tn = torch.cat(self.metrics['tn'][i][j])
                    
                    tp_sum = tp.sum(0, keepdim=True)
                    fp_sum = fp.sum(0, keepdim=True)
                    fn_sum = fn.sum(0, keepdim=True)
                    tn_sum = tn.sum(0, keepdim=True)
                    
                    metric_func = getattr(smp.metrics, metric_name)
                    scores[i, j] = metric_func(tp_sum, fp_sum, fn_sum, tn_sum)[0].cpu().detach()
                    mean_scores[i, j] = metric_func(tp, fp, fn, tn, reduction="micro").cpu().detach()
            
            results[metric_name] = scores.tolist()
            results[f"mean_{metric_name}"] = mean_scores.tolist()

        # Save results
        logdir = Path(self.logdir)
        results_filename = f"test_results_{self.test_dataset_name}.json"
        with open(logdir / results_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Test results saved to {logdir / results_filename}")


if __name__ == "__main__":
    from pathlib import Path

    # Sample configuration
    with open("configs/config19.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize datasets
    train_dataset = UltrasoundTrainDataset(
        Path(config["data"]["train_data_path"]),
        Path(config["data"]["train_annotations_path"]),
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
    )

    # Initialize the model
    model = TemporalSegmentationModel(
        encoder_name=config["model"]["encoder_name"],
        segmentation_model_name=config["model"]["segmentation_model_name"],
        num_classes=config["model"]["num_classes"],
        temporal_model=config["model"]["temporal_model"],
        image_size=tuple(config["model"]["image_size"]),
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
        batch_size=config["data"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
        logdir=None,
        alpha=config["model"]["alpha"],
        encoder_depth=config["model"]["encoder_depth"],
        temporal_depth=config["model"]["temporal_depth"],
        temporal_loss_weight=config["model"].get("temporal_loss_weight", 0.3),
        sparsity_weight=config["model"].get("sparsity_weight", 0.05),
    )

    # Show batch
    trainer.show_batch()
