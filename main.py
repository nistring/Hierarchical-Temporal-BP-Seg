import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from src.data_loader import UltrasoundTestDataset, UltrasoundTrainDataset
from src.model import TemporalSegmentationModel, SegmentationTrainer
import yaml
import torch
import argparse
from lightning.pytorch.strategies import DDPStrategy

class SaveConfigCallback(Callback):
    def __init__(self, config):
        self.config = config

    def on_train_start(self, trainer, pl_module):
        # Save the configuration file at the start of training
        config_path = os.path.join(trainer.log_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)

def main(config, best_model_path=None):
    """
    Main function to train the segmentation model.
    
    Args:
        config (dict): Configuration dictionary.
        best_model_path (str): Path to the best model checkpoint.
    """
    # Set visible CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    test = config.get("mode", "train") == "test"

    # Initialize datasets
    train_dataset = UltrasoundTrainDataset(
        Path(config["data"]["train_data_path"]),
        Path(config["data"]["train_annotations_path"]),
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
    )
    val_dataset = UltrasoundTrainDataset(
        Path(config["data"]["val_data_path"]),
        Path(config["data"]["val_annotations_path"]),
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
        train=False,
    )
    test_dataset = UltrasoundTestDataset(
        Path(config["data"]["val_data_path"]),
        Path(config["data"]["val_annotations_path"]),
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        batch_size=1,
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
        ),
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config["data"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"],
        logdir=Path(best_model_path).parent.parent if best_model_path else best_model_path,
        alpha=config["model"]["alpha"],  # Pass alpha to the trainer
        encoder_depth=config["model"]["encoder_depth"],  # Pass encoder_depth to the trainer
        temporal_depth=config["model"]["temporal_depth"],  # Pass temporal_depth to the trainer
    )

    # Set torch precision for matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Initialize callbacks
    save_config_callback = SaveConfigCallback(config)
    checkpoint_callback = ModelCheckpoint(
        monitor=config["logging"]["monitor"], mode=config["logging"]["mode"], save_last=True
    )

    # Initialize the trainer
    trainer = L.Trainer(
        strategy=DDPStrategy(),#find_unused_parameters=True),
        max_epochs=config["trainer"]["max_epochs"],
        devices=1 if test else config["trainer"]["gpus"],
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            save_config_callback,
        ],
        precision="bf16-mixed",
        sync_batchnorm=True,
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        use_distributed_sampler=False,
        logger=False if test else True
    )

    if test:
        # Load the best checkpoint before testing
        trainer.test(model, ckpt_path=best_model_path)
    else:
        # Start training
        trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the segmentation model.")
    parser.add_argument("--config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run: train or test.")
    parser.add_argument("--best_model_path", type=str, help="Path to the best model checkpoint for testing.")
    args = parser.parse_args()
    
    # Load configuration from file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
    config["mode"] = args.mode
    if args.best_model_path:
        config["best_model_path"] = args.best_model_path
    main(config, best_model_path=args.best_model_path)
