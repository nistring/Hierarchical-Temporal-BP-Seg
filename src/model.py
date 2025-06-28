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
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import sys
import yaml

sys.path.append("./")
from src.convgru import ConvGRU
from src.convlstm import ConvLSTM
from src.data_loader import UltrasoundTrainDataset, DistributedVideoSampler
from src.utils import post_processing
from src.losses import TemporalConsistencyLoss, SparsityLoss  # Import the new losses

class TemporalSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder_name,
        segmentation_model_name,
        num_classes,
        image_size,
        temporal_model="ConvLSTM",
        num_layers=1,
        encoder_depth=5,
        temporal_depth=1,
        freeze_encoder=False,
    ):
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
            freeze_encoder (bool): Whether to freeze the encoder weights. Default is False.
        """
        super().__init__()
        assert encoder_depth > temporal_depth, "Encoder depth should be greater than temporal depth."
        # Initialize the segmentation model from segmentation_models_pytorch
        self.num_classes = num_classes
        model = getattr(smp, segmentation_model_name)(
            encoder_name=encoder_name, encoder_weights="imagenet", classes=num_classes + 1, in_channels=1, encoder_depth=encoder_depth
        )
        
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.head = model.segmentation_head

        # Freeze encoder weights if specified
        self.freeze_encoder = freeze_encoder

        self.temporal_models = nn.ModuleList()
        h, w = image_size
        # Initialize the temporal models (ConvLSTM or ConvGRU) for each encoder output channel
        # base_depth = encoder_depth - temporal_depth
        base_depth = 2
        for i, out_channel in enumerate(self.encoder.out_channels[base_depth:base_depth + temporal_depth]):
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
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        features = [f.reshape(batch_size, seq_len, *f.shape[1:]) for f in features]

        if not self.temporal_models:  # Skip temporal processing if there are no temporal models
            temporal_outs = features
        else:
            temporal_outs = features[:2]
            
            if hidden_state is None:
                hidden_state = [None] * len(self.temporal_models)
            
            # Apply temporal models to the features
            for i, temporal_model in enumerate(self.temporal_models):
                temporal_out, hidden_state[i] = temporal_model(features[2+i], hidden_state[i])
                temporal_outs.append(temporal_out)

            temporal_outs.extend(features[2 + len(self.temporal_models):])

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
        temporal_loss_weight=1,  # Weight for temporal consistency loss
        sparsity_weight=0.05,     # Weight for sparsity loss
        ckpt_path=None,  # Path to load checkpoint
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
            temporal_loss_weight (float): Weight for temporal consistency loss. Default is 1.
            sparsity_weight (float): Weight for sparsity loss. Default is 0.05.
            ckpt_path (str): Path to load the model checkpoint. Default is None.
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
        self.alpha = alpha
        self.encoder_depth = encoder_depth
        self.temporal_depth = temporal_depth
        self.temporal_loss_weight = temporal_loss_weight
        self.sparsity_weight = sparsity_weight
        self.ckpt_path = ckpt_path

        # Define data augmentation transforms
        self.transform = v2.Compose(
            [
                v2.RandomApply([v2.RandomAffine(degrees=0, translate=(0.2, 0.1), scale=(0.85, 1.15), shear=0.15)]),
                v2.RandomApply([v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(3 / 4, 4 / 3))]),
                v2.RandomApply([v2.GaussianNoise()]),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2))]),
                v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.25)]),
                v2.RandomApply([v2.ElasticTransform()]),
            ]
        )

        # Initialize loss functions
        self.loss_tversky = smp.losses.TverskyLoss(mode="multiclass", from_logits=True, alpha=0.5, beta=0.5)
        self.loss_crossentropy = nn.CrossEntropyLoss()
        self.loss_temporal = TemporalConsistencyLoss()
        self.loss_sparsity = SparsityLoss()

    def show_batch(self, win_size=(80, 100), save=True):
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

        preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.image_size),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        # Apply data augmentation transforms
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            imgs[i], masks[i] = preprocess(img, tv_tensors.Mask(mask))
            imgs_aug[i], masks_aug[i] = self.transform(imgs[i], masks[i])  # apply transforms

        original_images = _to_vis(imgs, masks)
        augmented_images = _to_vis(imgs_aug, masks_aug)

        def _plot_images(images, filename):
            plt.figure(figsize=win_size)
            grid = torchvision.utils.make_grid(list(map(pil_to_tensor, images)), nrow=self.sequence_length)
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
        return images, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=DistributedVideoSampler(self.train_dataset, batch_size=self.batch_size),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=DistributedVideoSampler(self.val_dataset, batch_size=self.batch_size),
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.ckpt_path:
            return optimizer
        # Linear warmup scheduler
        scheduler = LinearLR(
            optimizer, 
            start_factor=0.1,  # Start at 10% of target LR
            end_factor=1.0,    # End at 100% of target LR
            total_iters=int(len(self.train_dataset) / self.truncated_bptt_steps / self.batch_size / self.trainer.num_devices * 5)  # Warmup for 5 epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update LR every step
                "frequency": 1,
            }
        }

    def compute_loss(self, batch, batch_idx, stage):
        images, targets = batch

        masks, self.hidden_state = self.model(images, self.hidden_state)

        # Keep the original shape for temporal consistency loss
        masks_original = masks

        masks = masks.reshape(-1, *masks.shape[2:])  # Flatten the batch and sequence dimensions
        targets = targets["masks"].reshape(-1, *targets["masks"].shape[2:]).long()  # Flatten the batch and sequence dimensions
        loss_tversky = self.loss_tversky(masks, targets)
        loss_crossentropy = self.loss_crossentropy(masks, targets)
        loss_temporal = self.loss_temporal(masks_original)
        loss_sparsity = self.loss_sparsity(masks)

        # Combine all losses
        loss = loss_tversky + self.alpha * loss_crossentropy + self.temporal_loss_weight * loss_temporal + self.sparsity_weight * loss_sparsity

        # Log each component of the loss
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss_tversky", loss_tversky, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss_crossentropy", loss_crossentropy, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss_temporal", loss_temporal, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss_sparsity", loss_sparsity, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if batch_idx % self.truncated_bptt_steps == 0:
            self.hidden_state = None
        elif self.temporal_depth > 0:
            # Detach the hidden state to prevent backpropagation through the entire sequence
            # This is important for truncated backpropagation
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
        # For grided results
        self.tp = [[[] for _ in range(self.truncated_bptt_steps)] for _ in range(self.truncated_bptt_steps)]
        self.fp = [[[] for _ in range(self.truncated_bptt_steps)] for _ in range(self.truncated_bptt_steps)]
        self.fn = [[[] for _ in range(self.truncated_bptt_steps)] for _ in range(self.truncated_bptt_steps)]
        self.tn = [[[] for _ in range(self.truncated_bptt_steps)] for _ in range(self.truncated_bptt_steps)]

        self.video_id = None
        # Get the base name of data_dir for the results filename
        self.test_dataset_name = Path(self.test_dataset.data_dir).name

    def test_step(self, batch):
        images, targets = batch
        if self.video_id != targets["video_id"]:
            self.video_id = targets["video_id"]
            self.hidden_states = []

        if len(self.hidden_states) < self.truncated_bptt_steps:
            self.hidden_states.append(None)

        targets["masks"] = F.one_hot(targets["masks"][0].long(), num_classes=self.model.num_classes+1)[..., 1:].permute(
            0, 3, 1, 2
        )  # Remove the background class
        
        def _process_masks_and_get_stats(masks, target_masks):
            masks = post_processing(masks)[:, 1:]  # Remove the background class
            # Convert masks to binary by taking argmax and using sum as values
            masks_argmax = torch.argmax(masks, dim=1, keepdim=True)
            masks_sum = torch.sum(masks, dim=1, keepdim=True)
            masks = torch.zeros_like(masks)
            masks.scatter_(1, masks_argmax, masks_sum)
            return smp.metrics.get_stats(masks, target_masks, mode="multilabel", threshold=0.5)
        
        for i, hidden_state in enumerate(self.hidden_states):
            masks, self.hidden_states[i] = self.model(images, hidden_state)
            tp, fp, fn, tn = _process_masks_and_get_stats(masks, targets["masks"])

            for j in range(i, len(self.hidden_states)):
                self.tp[i][j].append(tp)
                self.fp[i][j].append(fp)
                self.fn[i][j].append(fn)
                self.tn[i][j].append(tn)

    def on_test_epoch_end(self):
        # Compute the metrics for each segment
        iou_scores = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps, self.model.num_classes)
        f1_scores = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps, self.model.num_classes)
        precisions = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps, self.model.num_classes)
        sensitivities = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps, self.model.num_classes)

        mean_iou_score = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps)
        mean_f1_score = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps)
        mean_precision = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps)
        mean_sensitivity = torch.zeros(self.truncated_bptt_steps, self.truncated_bptt_steps)

        for i in range(self.truncated_bptt_steps):
            for j in range(i, self.truncated_bptt_steps):
                tp = torch.cat(self.tp[i][j])
                fp = torch.cat(self.fp[i][j])
                fn = torch.cat(self.fn[i][j])
                tn = torch.cat(self.tn[i][j])

                # seq = (tp + fn).sum(1, keepdim=True).bool()
                # tp = tp * seq
                # fp = fp * seq
                # fn = fn * seq
                # tn = tn * seq

                tp_sum = tp.sum(0, keepdim=True)
                fp_sum = fp.sum(0, keepdim=True)
                fn_sum = fn.sum(0, keepdim=True)
                tn_sum = tn.sum(0, keepdim=True)

                iou_scores[i, j] = smp.metrics.iou_score(tp_sum, fp_sum, fn_sum, tn_sum)[0].cpu().detach()
                f1_scores[i, j] = smp.metrics.f1_score(tp_sum, fp_sum, fn_sum, tn_sum)[0].cpu().detach()
                precisions[i, j] = smp.metrics.precision(tp_sum, fp_sum, fn_sum, tn_sum)[0].cpu().detach()
                sensitivities[i, j] = smp.metrics.sensitivity(tp_sum, fp_sum, fn_sum, tn_sum)[0].cpu().detach()

                mean_iou_score[i, j] = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").cpu().detach()
                mean_f1_score[i, j] = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").cpu().detach()
                mean_precision[i, j] = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").cpu().detach()
                mean_sensitivity[i, j] = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro").cpu().detach()

        results = {
            "iou_score": iou_scores.tolist(),
            "f1_score": f1_scores.tolist(),
            "precision": precisions.tolist(),
            "sensitivity": sensitivities.tolist(),
            "mean_iou_score": mean_iou_score.tolist(),
            "mean_f1_score": mean_f1_score.tolist(),
            "mean_precision": mean_precision.tolist(),
            "mean_sensitivity": mean_sensitivity.tolist(),
        }

        logdir = Path(self.logdir)
        # Use the test dataset name for the results file
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
        sequence_length=8,
        image_size=tuple(config["model"]["image_size"]),
        batch_size=10,
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
    )

    # Initialize the model and trainer
    model = SegmentationTrainer(
        TemporalSegmentationModel(
            encoder_name=config["model"]["encoder_name"],
            segmentation_model_name=config["model"]["segmentation_model_name"],
            num_classes=config["model"]["num_classes"],
            temporal_model=config["model"]["temporal_model"],
            image_size=tuple(config["model"]["image_size"]),
            encoder_depth=config["model"]["encoder_depth"],  # Pass encoder_depth
            temporal_depth=config["model"]["temporal_depth"],  # Pass temporal_depth
            freeze_encoder=config["model"].get("freeze_encoder", False),  # Pass freeze_encoder
        ),
        train_dataset,
        None,
        None,
        batch_size=10,
        learning_rate=config["model"]["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=8,
        image_size=tuple(config["model"]["image_size"]),
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
        logdir=None,
        alpha=config["model"]["alpha"],  # Pass alpha to the trainer
        encoder_depth=config["model"]["encoder_depth"],  # Pass encoder_depth to the trainer
        temporal_depth=config["model"]["temporal_depth"],  # Pass temporal_depth to the trainer
        temporal_loss_weight=config["model"].get("temporal_loss_weight", 0.3),  # Keep temporal_loss_weight
    )

    # Show batch
    model.show_batch()
