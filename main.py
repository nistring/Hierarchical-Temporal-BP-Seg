import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback, StochasticWeightAveraging
from src.data_loader import UltrasoundTestDataset, UltrasoundTrainDataset
from src.model import TemporalSegmentationModel, SegmentationTrainer
from src.utils import replace_gelu_with_relu
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
            f.write(f"Input shape: {x.shape})\n")
            f.write(f"Forward FLOPs: {fwd_flops:,}\nForward GFLOPs: {fwd_flops / 1e9:.2f}\n")


def _parse_train_basename(name: str):
    """Return (vendor, fold_token, family) from a train-COCO basename.

    vendor       : 'ge' | 'mindray' | 'sonosite' | None
    fold_token   : 'fold0' .. 'fold3' | None
    family       : 'sam2' | 'ultrasam' | None
    """
    import re
    family = "ultrasam" if "_ultrasam" in name else ("sam2" if "_sam2" in name else None)
    vendor = next((v for v in ("ge", "mindray", "sonosite") if name.startswith(v + "_")), None)
    m = re.search(r"fold(\d+)", name)
    fold_token = f"fold{m.group(1)}" if m else None
    return vendor, fold_token, family


def _resolve_label_family_auto(config):
    """Rewrite val/test ``annotations_path`` of ``auto`` to match the label
    family used at training time. Family (sam2 vs ultrasam) is inferred from
    the train annotations basename.

    Val derivation: ``<vendor>[_fold{k}]_val_<family>.json`` next to the train
    annotations file. The fold (if any) is preserved so a fold-trained model
    evaluates against the same fold's held-out 6 videos.

    Test derivation: the test ``data_path`` basename (e.g. ``ge_val`` or
    ``sonosite_val``) is combined with the trained family to produce
    ``<vendor_split>_<family>.json``.
    """
    data_cfg = config["data"]
    train_p = Path(data_cfg["train"]["annotations_path"])
    vendor, fold_token, family = _parse_train_basename(train_p.name)

    val_cfg = data_cfg.get("val", {})
    if val_cfg.get("annotations_path") == "auto":
        if family is None or vendor is None:
            raise ValueError(
                "data.val.annotations_path: auto needs train annotations whose "
                f"basename starts with a vendor and contains '_sam2' or "
                f"'_ultrasam'. Got: {train_p.name}"
            )
        prefix = f"{vendor}_{fold_token}" if fold_token else vendor
        val_cfg["annotations_path"] = str(
            train_p.with_name(f"{prefix}_val_{family}.json")
        )

    test_cfg = data_cfg.get("test", {})
    if test_cfg.get("annotations_path") == "auto":
        if family is None:
            raise ValueError(
                "data.test.annotations_path: auto requires '_sam2' or "
                f"'_ultrasam' in train annotations. Got: {train_p.name}"
            )
        vendor_split = Path(test_cfg["data_path"]).name
        test_cfg["annotations_path"] = str(
            train_p.with_name(f"{vendor_split}_{family}.json")
        )


def create_datasets(config):
    """Create train, validation, and test datasets."""
    _resolve_label_family_auto(config)
    data_cfg, model_cfg = config["data"], config["model"]

    train_dataset = UltrasoundTrainDataset(
        Path(data_cfg["train"]["data_path"]), Path(data_cfg["train"]["annotations_path"]),
        sequence_length=data_cfg["train"]["sequence_length"],
        image_size=tuple(model_cfg["image_size"]),
        batch_size=data_cfg["train"]["batch_size"],
        truncated_bptt_steps=data_cfg["train"]["truncated_bptt_steps"],
    )
    val_dataset = UltrasoundTrainDataset(
        Path(data_cfg["val"]["data_path"]), Path(data_cfg["val"]["annotations_path"]),
        sequence_length=data_cfg["val"]["sequence_length"],
        image_size=tuple(model_cfg["image_size"]),
        batch_size=data_cfg["val"]["batch_size"],
        truncated_bptt_steps=data_cfg["val"]["truncated_bptt_steps"],
        train=False
    )
    test_dataset = UltrasoundTestDataset(
        Path(data_cfg.get("test", {}).get("data_path", data_cfg["val"]["data_path"])), 
        Path(data_cfg.get("test", {}).get("annotations_path", data_cfg["val"]["annotations_path"])),
        sequence_length=100, 
        image_size=tuple(model_cfg["image_size"]), 
        batch_size=1
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
        "encoder_depth": model_cfg["encoder_depth"],
        "temporal_depth": model_cfg["temporal_depth"],
        "freeze_encoder": model_cfg.get("freeze_encoder", False),
        "num_layers": model_cfg.get("num_layers", 1),
        "kernel_size": tuple(model_cfg.get("kernel_size", [3, 3])),
        "dilation": model_cfg.get("dilation", 1),
        "conv_type": model_cfg.get("conv_type", "standard"),
        "encoder_weights": model_cfg.get("encoder_weights", "imagenet"),
        "use_hierarchical_fusion": model_cfg.get("use_hierarchical_fusion", True),
        **model_cfg.get("model_kwargs", {})
    }
    
    # R2.1-3 baseline dispatch: segmentation_model_name of form "baseline:<name>"
    # constructs a standalone per-frame baseline (no temporal fusion). See
    # src/baselines/ for the vendored models.
    if model_cfg["segmentation_model_name"].startswith("baseline:"):
        from src.baselines import make_baseline
        model = make_baseline(
            name=model_cfg["segmentation_model_name"].split(":", 1)[1],
            num_classes=model_cfg["num_classes"] + 1,
            input_channels=1,
            image_size=tuple(model_cfg["image_size"]),
        )
    else:
        model = TemporalSegmentationModel(**model_config)
    if model_cfg.get("use_relu", False):
        model = replace_gelu_with_relu(model)
    if config["trainer"].get("ckpt_path"):
        model = load_model(model, config["trainer"]["ckpt_path"])

    # Create trainer module
    lit_module = SegmentationTrainer(
        model, train_dataset, val_dataset, test_dataset,
        batch_size=config["data"]["train"]["batch_size"],
        learning_rate=model_cfg["learning_rate"],
        num_workers=config["data"]["num_workers"],
        sequence_length=config["data"]["train"]["sequence_length"],
        image_size=tuple(model_cfg["image_size"]),
        truncated_bptt_steps=config["data"]["train"]["truncated_bptt_steps"],
        logdir=Path(best_model_path).parent.parent if best_model_path else None,
        ce_weight=model_cfg["ce_weight"],
        temporal_depth=model_cfg["temporal_depth"],
        temporal_loss_weight=model_cfg.get("temporal_loss_weight", 0.3),
        negative_weight=model_cfg.get("negative_weight", 100),
        positive_weight=model_cfg.get("positive_weight", 10),
        exclusion_weight=model_cfg.get("exclusion_weight", 0.05),
        exclusion_groups=model_cfg.get("exclusion_groups"),
        ckpt_path=bool(config["trainer"].get("ckpt_path")),
        ce_class_weights=model_cfg.get("ce_class_weights"),
        manual_annotations_path=config.get("manual_annotations_path"),
    )

    # Setup trainer
    trainer_cfg = config["trainer"]
    # SWA params can be overridden under trainer.swa: {epoch_start, lrs,
    # annealing_epochs}. Defaults preserve the original 20-epoch baseline:
    # last 7 epochs of SWA, swa_lrs = 0.5 × initial LR, 4 annealing epochs.
    swa_cfg = trainer_cfg.get("swa", {}) or {}
    swa_epoch_start = swa_cfg.get("epoch_start", trainer_cfg["max_epochs"] - 7)
    swa_lrs = swa_cfg.get("lrs", 0.5 * model_cfg["learning_rate"])
    swa_annealing = swa_cfg.get("annealing_epochs", 4)
    callbacks = [
        ModelCheckpoint(monitor=config["logging"]["monitor"], mode=config["logging"]["mode"], save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
        SaveConfigCallback(config, model_config),
        StochasticWeightAveraging(
            swa_epoch_start=swa_epoch_start,
            swa_lrs=swa_lrs,
            annealing_epochs=swa_annealing,
        )
    ]

    find_unused_parameters = False
    seg_name = config["model"]["segmentation_model_name"].lower()
    if "deeplab" in seg_name or "rolling_unet" in seg_name:
        find_unused_parameters = True
    
    version_name = config["config_file"].split("/")[-1].split(".")[0]
    if not test_mode:
        log_dir = Path("./lightning_logs")
        if (log_dir / version_name).exists():
            i = 1
            while (log_dir / f"{version_name}_{i}").exists():
                i += 1
            version_name = f"{version_name}_{i}"

    trainer = L.Trainer(
        strategy=DDPStrategy(static_graph=False, gradient_as_bucket_view=True, find_unused_parameters=find_unused_parameters),
        max_epochs=trainer_cfg["max_epochs"],
        devices=trainer_cfg["gpus"],
        callbacks=callbacks,
        precision="bf16-mixed",
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=False if test_mode else TensorBoardLogger(save_dir="./", version=version_name),
        gradient_clip_val=1.0,
    )

    if test_mode:
        # weights_only=False: our checkpoints contain pickled objects (e.g.
        # `getattr` references in saved hyperparams) that torch>=2.6's
        # default `weights_only=True` rejects.
        trainer.test(lit_module, ckpt_path=best_model_path, weights_only=False)
    else:
        trainer.fit(lit_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the segmentation model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--best_model_path", type=str, help="Path to the best model checkpoint for testing.")
    parser.add_argument("--test_data_path", type=str, help="Path to the test data directory.")
    parser.add_argument("--test_annotations_path", type=str, help="Path to the test annotations file.")
    parser.add_argument("--manual_annotations_path", type=str,
                        help="Optional sparse-polygon manual COCO. When set, "
                             "test_step also computes per-(frame, class) "
                             "metrics restricted to the classes that have a "
                             "manual label on each frame.")
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

    if isinstance(args.gpu, int):
        config["trainer"]["gpus"] = (args.gpu,)
    
    if args.test_data_path:
        # --test_annotations_path is optional now: when omitted, derive it from
        # the train annotations' label family (sam2 vs ultrasam) by setting
        # annotations_path to 'auto' and letting _resolve_label_family_auto fill it.
        config["data"].update({
            "test": {
                "data_path": args.test_data_path,
                "annotations_path": args.test_annotations_path or "auto",
            }
        })
    elif args.test_annotations_path:
        raise ValueError("--test_annotations_path requires --test_data_path.")

    if args.manual_annotations_path:
        config["manual_annotations_path"] = args.manual_annotations_path
        
    main(config, best_model_path=args.best_model_path)
