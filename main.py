import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, StochasticWeightAveraging, ModelSummary
from src.data_loader import UltrasoundTestDataset, UltrasoundTrainDataset
from src.model import TemporalSegmentationModel, SegmentationTrainer
import yaml
import torch
import argparse
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from src.utils import load_model


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
        Path(config["data"]["test_data_path"]),
        Path(config["data"]["test_annotations_path"]),
        sequence_length=50, # Use a fixed sequence length for testing, 5 seconds
        image_size=tuple(config["model"]["image_size"]),
        batch_size=1,
    )

    # Initialize the model and trainer
    model = TemporalSegmentationModel(
        encoder_name=config["model"]["encoder_name"],
        segmentation_model_name=config["model"]["segmentation_model_name"],
        num_classes=config["model"]["num_classes"],
        temporal_model=config["model"]["temporal_model"],
        image_size=tuple(config["model"]["image_size"]),
        encoder_depth=config["model"]["encoder_depth"],
        temporal_depth=config["model"]["temporal_depth"],
        freeze_encoder=config["model"].get("freeze_encoder", False),  # Get freeze_encoder from config
    )
    if config["trainer"].get("ckpt_path", False):
        model = load_model(model, config["trainer"]["ckpt_path"])

    lit_module = SegmentationTrainer(
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config["data"]["batch_size"],
        learning_rate=config["model"]["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=config["model"]["sequence_length"],
        image_size=tuple(config["model"]["image_size"]),
        truncated_bptt_steps=config["model"]["truncated_bptt_steps"] if not test else 12, # Use 12 for testing
        logdir=Path(best_model_path).parent.parent if best_model_path else best_model_path,
        alpha=config["model"]["alpha"],
        encoder_depth=config["model"]["encoder_depth"],
        temporal_depth=config["model"]["temporal_depth"],
        temporal_loss_weight=config["model"].get("temporal_loss_weight", 0.3),
        sparsity_weight= config["model"].get("sparsity_weight", 0.0),
        ckpt_path=True if config["trainer"].get("ckpt_path", False) else False,
    )

    # Set torch precision for matrix multiplication
    torch.set_float32_matmul_precision("high")

    callbacks=[
        ModelSummary(max_depth=3),
        ModelCheckpoint(monitor=config["logging"]["monitor"], mode=config["logging"]["mode"], save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
        SaveConfigCallback(config),
        StochasticWeightAveraging(
            swa_epoch_start=config["trainer"]["max_epochs"] - 10,
            swa_lrs=0.5 * config["model"]["learning_rate"],
            annealing_epochs=5
        )
    ]

    # Initialize the trainer
    trainer = L.Trainer(
        # strategy=DDPStrategy(),
        strategy="ddp_find_unused_parameters_true",
        max_epochs=config["trainer"]["max_epochs"],
        devices=config["trainer"]["gpus"],
        callbacks=callbacks,
        precision="bf16-mixed",
        sync_batchnorm=True,
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        use_distributed_sampler=False,
        logger=False if test else TensorBoardLogger(save_dir="./",version=config["config_file"].split("/")[-1].split(".")[0]),
    )

    if test:
        # Load the best checkpoint before testing
        trainer.test(lit_module, ckpt_path=best_model_path)
    else:
        # Start training
        trainer.fit(lit_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the segmentation model.")
    parser.add_argument("--config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run: train or test.")
    parser.add_argument("--best_model_path", type=str, help="Path to the best model checkpoint for testing.")
    parser.add_argument("--test_data_path", type=str, help="Path to the test data directory.")
    parser.add_argument("--test_annotations_path", type=str, help="Path to the test annotations file.")
    parser.add_argument("--gpu", type=int, help="GPU device ID to use. Only for test step.")
    args = parser.parse_args()
    
    # Load configuration from file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    if args.gpu:
        config["trainer"]["gpus"] = (args.gpu,)
    # Set the mode and best model path in the config
    
    config["config_file"] = args.config_file
    config["mode"] = args.mode
    if args.best_model_path:
        config["best_model_path"] = args.best_model_path
    
    # Override test paths if provided
    if args.test_data_path and args.test_annotations_path:
        config["data"]["test_data_path"] = args.test_data_path
        config["data"]["test_annotations_path"] = args.test_annotations_path
    elif args.test_data_path or args.test_annotations_path:
        raise ValueError("Both test_data_path and test_annotations_path must be provided.")
    else:
        config["data"]["test_data_path"] = config["data"]["val_data_path"]
        config["data"]["test_annotations_path"] = config["data"]["val_annotations_path"]
        
    main(config, best_model_path=args.best_model_path)
