import json
from pathlib import Path

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import tv_tensors
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import sys

sys.path.append("./")
from src.convgru import ConvGRU
from src.convlstm import ConvLSTM
from src.data_loader import UltrasoundTrainDataset, DistributedVideoSampler
from src.utils import convert_to_coco_format


class TemporalSegmentationModel(nn.Module):
    def __init__(self, encoder_name, segmentation_model_name, num_classes, image_size, temporal_model="ConvLSTM", num_layers=1, encoder_depth=5, temporal_depth=1):
        """
        Initialize the TemporalSegmentationModel.

        Args:
            encoder_name (str): Name of the encoder.
            segmentation_model_name (str): Name of the segmentation model.
            num_classes (int): Number of classes.
            image_size (tuple): Size of the input image.
            temporal_model (str): Type of temporal model. Default is "ConvLSTM".
            num_layers (int): Number of layers in the temporal model. Default is 1.
            encoder_depth (int): Depth of the encoder. Default is 5.
            temporal_depth (int): Depth of the temporal model. Default is 1.
        """
        super().__init__()
        assert encoder_depth >= temporal_depth, "Encoder depth should be greater than or equal to temporal depth."
        # Initialize the segmentation model from segmentation_models_pytorch
        model = getattr(smp, segmentation_model_name)(
            encoder_name=encoder_name, encoder_weights="imagenet", classes=num_classes + 1, in_channels=1, encoder_depth=encoder_depth
        )
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.head = model.segmentation_head

        self.temporal_models = nn.ModuleList()
        h, w = image_size
        # Initialize the temporal models (ConvLSTM or ConvGRU) for each encoder output channel
        base_depth = encoder_depth - temporal_depth
        if "mit" in encoder_name:
            base_depth += 1
        for i, out_channel in enumerate(self.encoder.out_channels[base_depth:]):
            if temporal_model == "ConvLSTM":
                self.temporal_models.append(
                    ConvLSTM(
                        input_size=(h // (2 ** (i + base_depth)), w // (2 ** (i + base_depth))),
                        input_dim=out_channel,
                        hidden_dim=out_channel,
                        kernel_size=(3, 3),
                        num_layers=num_layers,
                        batch_first=True,
                    )
                )
            elif temporal_model == "ConvGRU":
                self.temporal_models.append(
                    ConvGRU(
                        input_size=(h // (2 ** (i + base_depth)), w // (2 ** (i + base_depth))),
                        input_dim=out_channel,
                        hidden_dim=out_channel,
                        kernel_size=(3, 3),
                        num_layers=num_layers,
                        batch_first=True,
                    )
                )
            else:
                raise ValueError("Unsupported temporal model type. Choose from 'ConvLSTM' or 'ConvGRU'.")

    def forward(self, x, hidden_state=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            hidden_state (torch.Tensor, optional): Hidden state for the temporal model.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Updated hidden state.
        """
        batch_size, seq_len, c, h, w = x.size()
        x = x.reshape(batch_size * seq_len, c, h, w)

        # Extract features using the encoder
        features = self.encoder(x)
        features = [f.reshape(batch_size, seq_len, *f.shape[1:]) for f in features]

        temporal_outs = features[:-len(self.temporal_models)]

        if hidden_state is None:
            hidden_state = [None] * len(self.temporal_models)

        # Apply temporal models to the features
        for i, temporal_model in enumerate(self.temporal_models):
            temporal_out, hidden_state[i] = temporal_model(features[i - len(self.temporal_models)], hidden_state[i])
            temporal_outs.append(temporal_out)
        temporal_outs = [f.reshape(batch_size * seq_len, *f.shape[2:]) for f in temporal_outs]

        # Decode the features to get the segmentation output
        x = self.decoder(temporal_outs)
        x = self.head(x)
        x = x.reshape(batch_size, seq_len, *x.shape[1:])

        return x, hidden_state

class SegmentationTrainer(L.LightningModule):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        learning_rate,
        num_workers,
        sequence_length,
        image_size,
        truncated_bptt_steps,
        logdir=None,
        alpha=0.5,  # Add alpha parameter
        encoder_depth=5,  # Add encoder_depth parameter
        temporal_depth=1,  # Add temporal_depth parameter
    ):
        """
        Initialize the SegmentationTrainer.

        Args:
            model (nn.Module): Segmentation model.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Dataset): Test dataset.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            num_workers (int): Number of workers for data loading.
            sequence_length (int): Sequence length.
            image_size (tuple): Size of the input image.
            truncated_bptt_steps (int): Truncated backpropagation through time steps.
            logdir (str): Directory to save the logs. Default is None.
            alpha (float): Coefficient for CrossEntropy loss. Default is 0.5.
            encoder_depth (int): Depth of the encoder. Default is 5.
            temporal_depth (int): Depth of the temporal model. Default is 1.
        """
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
        self.alpha = alpha  # Set alpha
        self.encoder_depth = encoder_depth  # Set encoder_depth
        self.temporal_depth = temporal_depth  # Set temporal_depth

        # Define data augmentation transforms
        self.transform = v2.Compose(
            [
                v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.15, 0.05), scale=(0.9, 1.1), shear=0.1)]),
                v2.RandomApply([v2.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.8, 1.25))]),
                v2.RandomApply([v2.GaussianNoise()]),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))]),
                v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)]),
            ]
        )

        self.scale = v2.ToDtype(torch.float32, scale=True)
        self.loss_tversky = smp.losses.TverskyLoss(mode="multiclass", from_logits=True, alpha=0.5, beta=0.5)
        self.loss_crossentropy = nn.CrossEntropyLoss()
        self.alpha = alpha  # Coefficient for CrossEntropy loss

    def show_batch(self, win_size=(50, 25), save=True):
        """
        Show a batch of images and masks.

        Args:
            win_size (tuple): Window size for displaying images.
            save (bool): Whether to save the images. Default is True.
        """
        def _to_vis(imgs, masks):
            imgs = imgs.reshape(-1, *imgs.shape[2:]).cpu()  # Flatten the batch and sequence dimensions
            masks = masks.reshape(-1, 1, *targets["masks"].shape[2:]).bool().float().cpu()
            vis_images = []
            for img, mask in zip(imgs, masks):
                img = to_pil_image(img)
                mask = to_pil_image(mask)  # Normalize mask to [0, 1] for visualization
                img.paste(mask, (0, 0), mask)
                vis_images.append(img)
            return vis_images

        # Create a DataLoader for the training dataset
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        imgs, targets = next(iter(train_dataloader))
        masks = targets["masks"]
        imgs_aug, masks_aug = imgs.clone(), masks.clone()
        
        # Apply data augmentation transforms
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            imgs_aug[i], masks_aug[i] = self.transform(img, tv_tensors.Mask(mask))  # apply transforms
        imgs = self.scale(imgs)
        imgs_aug = self.scale(imgs_aug)

        original_images = _to_vis(imgs, masks)
        augmented_images = _to_vis(imgs_aug, masks_aug)

        def _plot_images(images, filename):
            plt.figure(figsize=win_size)
            grid = torchvision.utils.make_grid([torchvision.transforms.ToTensor()(img) for img in images], nrow=self.sequence_length)
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            if save:
                plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            else:
                plt.show()

        _plot_images(original_images, "original.png")
        _plot_images(augmented_images, "augmented.png")

    def on_after_batch_transfer(self, batch, batch_idx):
        images, targets = batch
        if self.trainer.training:
            images, targets["masks"] = self.transform(images, tv_tensors.Mask(targets["masks"]))  # perform GPU/Batched data augmentation
            targets["masks"] = torch.Tensor(targets["masks"])
        images = self.scale(images)  # Scale images
        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=DistributedVideoSampler(self.train_dataset),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=DistributedVideoSampler(self.val_dataset),
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def compute_loss(self, batch, batch_idx, stage):
        images, targets = batch

        masks, self.hidden_state = self.model(images, self.hidden_state)

        masks = masks.reshape(-1, *masks.shape[2:])  # Flatten the batch and sequence dimensions
        targets = targets["masks"].reshape(-1, *targets["masks"].shape[2:])  # Flatten the batch and sequence dimensions
        
        loss_tversky = self.loss_tversky(masks, targets.long())
        loss_crossentropy = self.loss_crossentropy(masks, targets.long())
        loss = loss_tversky + self.alpha * loss_crossentropy

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % self.truncated_bptt_steps == 0:
            self.hidden_state = None
        else:
            self.hidden_state = [[h.detach() for h in hidden_state] for hidden_state in self.hidden_state]

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "train")
    
    def on_train_epoch_end(self):
        self.train_dataset._create_img_list()

    def on_validation_epoch_start(self):
        self.hidden_state = None

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "val")

    def on_test_start(self):
        self.coco_gt = self.test_dataset.coco
        self.coco_dt = []
        self.tp, self.fp, self.fn, self.tn = [], [], [], []
        self.video_id = None

    def test_step(self, batch):
        images, targets = batch

        if self.video_id != targets["video_id"]:
            self.video_id = targets["video_id"]
            self.hidden_state = None

        masks, self.hidden_state = self.model(images, self.hidden_state)
        targets["masks"] = F.one_hot(targets["masks"][0].long(), num_classes=masks.shape[2])[..., 1:].permute(0, 3, 1, 2)  # Remove the background class
        masks = torch.nn.Softmax(dim=1)(masks[0])[:, 1:] # .cpu().detach().numpy()  # Remove the background class

        tp, fp, fn, tn = smp.metrics.get_stats(masks, targets["masks"], mode='multilabel', threshold=0.5)
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tn.append(tn)

    def on_test_epoch_end(self):
        self.tp = torch.cat(self.tp)
        self.fp = torch.cat(self.fp)
        self.fn = torch.cat(self.fn)
        self.tn = torch.cat(self.tn)
        seq_obj_appear = (self.tp + self.fn).bool()

        iou_score = smp.metrics.iou_score(self.tp, self.fp, self.fn, self.tn)
        f1_score = smp.metrics.f1_score(self.tp, self.fp, self.fn, self.tn)
        overall_score = 0.5 * (iou_score + f1_score)
        seq_obj_appear_sum = seq_obj_appear.sum(0)
        
        iou_score = ((iou_score * seq_obj_appear).sum(0) / seq_obj_appear_sum).cpu().detach()
        f1_score = ((f1_score * seq_obj_appear).sum(0) / seq_obj_appear_sum).cpu().detach()
        overall_score = ((overall_score * seq_obj_appear).sum(0) / seq_obj_appear_sum).cpu().detach()

        mean_iou_score = iou_score.mean()
        mean_f1_score = f1_score.mean()
        mean_overall_score = overall_score.mean()

        results = {
            "iou_score": iou_score.tolist(),
            "f1_score": f1_score.tolist(),
            "overall_score": overall_score.tolist(),
            "mean_iou_score": mean_iou_score.mean().item(),
            "mean_f1_score": mean_f1_score.mean().item(),
            "mean_overall_score": mean_overall_score.mean().item(),
        }

        logdir = Path(self.logdir)
        with open(logdir / "test_results.json", "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    from pathlib import Path

    # Sample configuration
    config = {
        "model": {
            "encoder_name": "resnet34",
            "segmentation_model_name": "Unet",
            "num_classes": 10,
            "temporal_model": "ConvGRU",
            "sequence_length": 10,
            "image_size": [480, 640],
            "truncated_bptt_steps": 10,
        },
        "data": {
            "train_data_path": "./data/SUIT/images/train",
            "train_annotations_path": "./data/SUIT/coco_annotations/train_updated.json",
            "batch_size": 8,
            "num_workers": 12,
        },
    }

    # Create datasets
    train_dataset = UltrasoundTrainDataset(
        Path(config["data"]["train_data_path"]),
        Path(config["data"]["train_annotations_path"]),
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
    )

    # Create model
    model = TemporalSegmentationModel(
        encoder_name=config["model"]["encoder_name"],
        segmentation_model_name=config["model"]["segmentation_model_name"],
        num_classes=config["model"]["num_classes"],
        temporal_model=config["model"]["temporal_model"],
        image_size=tuple(config["model"]["image_size"]),
    )

    # Print model size
    model.print_model_size()

    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        test_dataset=None,
        batch_size=config["data"]["batch_size"],
        learning_rate=1e-3,
        num_workers=config["data"]["num_workers"],
        image_size=tuple(config["model"]["image_size"]),
        sequence_length=config["model"]["sequence_length"],
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
    )

    # Show batch
    trainer.show_batch()