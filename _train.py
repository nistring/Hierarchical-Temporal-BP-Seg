import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from src.data_loader import UltrasoundDataset, UltrasoundTestDataset, UltrasoundTrainDataset
from src.model import TemporalSegmentationModel, SegmentationTrainer
import yaml
import torch
import argparse
from lightning.pytorch.strategies import DDPStrategy

class SaveConfigCallback(Callback):
    def __init__(self, config):
        self.config = config

    def on_train_start(self, trainer, pl_module):
        config_path = os.path.join(trainer.log_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)

def main(config_file):
    """
    Main function to train the segmentation model.
    
    Args:
        config_file (str): Path to the configuration file.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

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

    model = SegmentationTrainer(
        TemporalSegmentationModel(
            encoder_name=config["model"]["encoder_name"],
            segmentation_model_name=config["model"]["segmentation_model_name"],
            num_classes=config["model"]["num_classes"],
            temporal_model=config["model"]["temporal_model"],
            image_size=tuple(config["model"]["image_size"]),
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
    )

    torch.set_float32_matmul_precision("high")

    save_config_callback = SaveConfigCallback(config)

    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config["trainer"]["max_epochs"],
        devices=config["trainer"]["gpus"],
        callbacks=[
            ModelCheckpoint(
                monitor=config["logging"]["monitor"], mode=config["logging"]["mode"], save_top_k=config["logging"]["save_top_k"]
            ),
            LearningRateMonitor(logging_interval="step"),
            save_config_callback,
        ],
        precision="bf16-mixed",
        sync_batchnorm=True,
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        use_distributed_sampler=False,
    )

    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the segmentation model.")
    parser.add_argument("--config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config_file)
