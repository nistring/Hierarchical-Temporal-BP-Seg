import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, StochasticWeightAveraging
from src.data_loader import UltrasoundTestDataset, UltrasoundTrainDataset
from src.model import TemporalSegmentationModel, SegmentationTrainer
import yaml
import torch
import argparse
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from src.utils import load_model
from lightning.pytorch.utilities.model_summary import summarize
from lightning.pytorch.utilities import measure_flops


class SaveConfigCallback(Callback):
    def __init__(self, config, model_config):
        self.config = config
        self.model_config = model_config

    def on_train_start(self, trainer, pl_module):
        self._save_config(trainer.log_dir)
        self._save_model_info(trainer, pl_module)
    
    def _save_config(self, log_dir):
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
    
    def _save_model_info(self, trainer, pl_module):
        cfg = self.config["model"]
        with torch.no_grad():
            x = torch.randn(1, 1, 1, *cfg["image_size"], device=pl_module.device)
            fwd_flops = measure_flops(pl_module.model, lambda: pl_module.model(x))
        
        info_path = os.path.join(trainer.log_dir, "model_info.txt")
        with open(info_path, "w") as f:
            f.write("Model Summary and FLOPs Information\n" + "=" * 50 + "\n\n")
            f.write(str(summarize(pl_module, max_depth=3)))
            f.write(f"\n\nFLOPs Information:\n" + "-" * 20 + "\n")
            f.write(f"Input shape: (1, {cfg['sequence_length']}, 3, {cfg['image_size'][0]}, {cfg['image_size'][1]})\n")
            f.write(f"Forward FLOPs: {fwd_flops:,}\nForward GFLOPs: {fwd_flops / 1e9:.2f}\n")


def create_datasets(config):
    """Create train, validation, and test datasets."""
    data_cfg, model_cfg = config["data"], config["model"]
    common_params = {
        "sequence_length": model_cfg["sequence_length"],
        "image_size": tuple(model_cfg["image_size"]),
        "batch_size": data_cfg["batch_size"],
    }
    
    train_dataset = UltrasoundTrainDataset(
        Path(data_cfg["train_data_path"]), Path(data_cfg["train_annotations_path"]),
        truncated_bptt_steps=model_cfg["truncated_bptt_steps"], **common_params
    )
    val_dataset = UltrasoundTrainDataset(
        Path(data_cfg["val_data_path"]), Path(data_cfg["val_annotations_path"]),
        truncated_bptt_steps=model_cfg["truncated_bptt_steps"], train=False, **common_params
    )
    test_dataset = UltrasoundTestDataset(
        Path(data_cfg["test_data_path"]), Path(data_cfg["test_annotations_path"]),
        sequence_length=model_cfg["sequence_length"], image_size=tuple(model_cfg["image_size"]), batch_size=1
    )
    
    return train_dataset, val_dataset, test_dataset


def main(config, best_model_path=None):
    test_mode = config.get("mode") == "test"
    train_dataset, val_dataset, test_dataset = create_datasets(config)

    # Create model
    model_cfg = config["model"]
    model_config = {
        "encoder_name": model_cfg["encoder_name"],
        "segmentation_model_name": model_cfg["segmentation_model_name"],
        "num_classes": model_cfg["num_classes"],
        "temporal_model": model_cfg["temporal_model"],
        "image_size": tuple(model_cfg["image_size"]),
        "encoder_depth": model_cfg["encoder_depth"],
        "temporal_depth": model_cfg["temporal_depth"],
        "freeze_encoder": model_cfg.get("freeze_encoder", False),
        "num_layers": model_cfg.get("num_layers", 1),
        "kernel_size": tuple(model_cfg.get("kernel_size", [3, 3])),
        "dilation": model_cfg.get("dilation", 1),
        **model_cfg.get("model_kwargs", {})
    }
    
    model = TemporalSegmentationModel(**model_config)
    if config["trainer"].get("ckpt_path"):
        model = load_model(model, config["trainer"]["ckpt_path"])

    # Create trainer module
    lit_module = SegmentationTrainer(
        model, train_dataset, val_dataset, test_dataset,
        batch_size=config["data"]["batch_size"],
        learning_rate=model_cfg["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=model_cfg["sequence_length"],
        image_size=tuple(model_cfg["image_size"]),
        truncated_bptt_steps=12 if test_mode else model_cfg["truncated_bptt_steps"],
        logdir=Path(best_model_path).parent.parent if best_model_path else None,
        alpha=model_cfg["alpha"],
        encoder_depth=model_cfg["encoder_depth"],
        temporal_depth=model_cfg["temporal_depth"],
        temporal_loss_weight=model_cfg.get("temporal_loss_weight", 0.3),
        sparsity_weight=model_cfg.get("sparsity_weight", 0.0),
        ckpt_path=bool(config["trainer"].get("ckpt_path")),
    )

    torch.set_float32_matmul_precision("high")

    # Setup trainer
    trainer_cfg = config["trainer"]
    callbacks = [
        ModelCheckpoint(monitor=config["logging"]["monitor"], mode=config["logging"]["mode"], save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
        SaveConfigCallback(config, model_config),
        StochasticWeightAveraging(
            swa_epoch_start=trainer_cfg["max_epochs"] - 7,
            swa_lrs=0.5 * model_cfg["learning_rate"],
            annealing_epochs=4
        )
    ]

    trainer = L.Trainer(
        strategy=DDPStrategy(static_graph=False, gradient_as_bucket_view=True, find_unused_parameters=False),
        max_epochs=trainer_cfg["max_epochs"],
        devices=trainer_cfg["gpus"],
        callbacks=callbacks,
        precision="bf16-mixed",
        sync_batchnorm=True,
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        use_distributed_sampler=False,
        logger=False if test_mode else TensorBoardLogger(save_dir="./", version=config["config_file"].split("/")[-1].split(".")[0]),
    )

    if test_mode:
        trainer.test(lit_module, ckpt_path=best_model_path)
    else:
        trainer.fit(lit_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the segmentation model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--best_model_path", type=str, help="Path to the best model checkpoint for testing.")
    parser.add_argument("--test_data_path", type=str, help="Path to the test data directory.")
    parser.add_argument("--test_annotations_path", type=str, help="Path to the test annotations file.")
    parser.add_argument("--gpu", type=int, help="GPU device ID to use.")
    args = parser.parse_args()
    
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Apply command line overrides
    config.update({
        "config_file": args.config_file,
        "mode": args.mode,
        "best_model_path": args.best_model_path
    })
    
    if args.gpu:
        config["trainer"]["gpus"] = (args.gpu,)
    
    if args.test_data_path and args.test_annotations_path:
        config["data"].update({
            "test_data_path": args.test_data_path,
            "test_annotations_path": args.test_annotations_path
        })
    elif args.test_data_path or args.test_annotations_path:
        raise ValueError("Both test_data_path and test_annotations_path must be provided.")
    else:
        config["data"].update({
            "test_data_path": config["data"]["val_data_path"],
            "test_annotations_path": config["data"]["val_annotations_path"]
        })
        
    main(config, best_model_path=args.best_model_path)
