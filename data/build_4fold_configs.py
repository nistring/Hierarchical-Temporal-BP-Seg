"""Materialize all 4-fold R1.3 configs from a single hyperparameter source.

Emits, under ``configs/r3_4fold/``:
  - ``{vendor}_fold{k}_finetune_{labels}.yaml``    : supervised fine-tune on
                                                     the fold's 18 train videos
  - ``{vendor}_fold{k}_ssl_{labels}.yaml``         : SSL train (works for both
                                                     round 1 and round 2 — the
                                                     round is encoded in the
                                                     ssl train COCO referenced;
                                                     re-run data/build_4fold_ssl
                                                     with the desired --round
                                                     before each training)

The two stages share **all** hyperparameters except ``data.train.annotations_path``
and ``data.train.data_path``. SSL pseudo-label generation thresholds are not
in this config — they live in the data builder. See exp/run_4fold_ssl.sh for
the recommended thresholds (per-class softmax conf 0.60/0.65 hard-class mix,
frame-entropy gate at 1.5).

Usage:
    PYTHONPATH=. python data/build_4fold_configs.py
"""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "r3_4fold"


PRETRAIN_CKPT = {
    # Both source checkpoints are unweighted: seq50 (sam2) has ce_class_weights
    # commented out; r15_noweights is the unweighted twin of seq50_ultrasam.
    "sam2":      "lightning_logs/seq50/checkpoints/last.ckpt",
    "ultrasam":  "lightning_logs/r15_noweights_ultrasam_seed0/checkpoints/last.ckpt",
}

# Shared model hyperparams across all configs.
MODEL_BLOCK = {
    "encoder_name": "mit_b0",
    "segmentation_model_name": "Segformer",
    "temporal_model": "ConvLSTM",
    "conv_type": "depthwise",
    "num_classes": 8,
    "freeze_encoder": False,
    "num_layers": 1,
    "kernel_size": [3, 3],
    "dilation": 1,
    "encoder_depth": 5,
    "temporal_depth": 4,
    "model_kwargs": {"decoder_segmentation_channels": 256},
    "learning_rate": 1.0e-4,
    "image_size": [416, 416],
    "ce_weight": 0.5,
    "temporal_loss_weight": 5,
    "exclusion_weight": 5,
    "positive_weight": 5,
    "negative_weight": 5,
    # Unweighted CE (R1.3 unweighted retrain, 2026-05-26): ce_class_weights omitted
    # so the cross-device runs match the unweighted headline end-to-end.
}

TRAINER_BLOCK = {
    "max_epochs": 3,
    "gpus": [0, 1, 2, 3],
    # ckpt_path filled per-config based on labels family
    "swa": {
        "epoch_start": 0.0001,
        "lrs": 1.0e-5,
        "annealing_epochs": 3,
    },
}

LOGGING_BLOCK = {"monitor": "val_loss", "mode": "min"}


def _data_block(vendor: str, fold: int, labels: str, stage: str) -> dict:
    """Return the `data:` block for one (vendor, fold, labels, stage)."""
    if stage == "finetune":
        train_data_path = "./data/SUIT/images/"
        train_ann_path = (f"./data/SUIT/coco_annotations/"
                          f"{vendor}_fold{fold}_train_{labels}.json")
    elif stage == "ssl":
        # SSL COCO file_names are also rooted at data/SUIT/images/ (see
        # data/build_4fold_ssl.py — labeled videos keep their {vendor}_{train,val}/
        # prefix, pseudo entries get a {vendor}_pseudo/ prefix).
        train_data_path = "./data/SUIT/images/"
        train_ann_path = (f"./data/SUIT/coco_annotations/"
                          f"{vendor}_fold{fold}_ssl_{labels}_train.json")
    else:
        raise ValueError(stage)

    return {
        "train": {
            "data_path": train_data_path,
            "annotations_path": train_ann_path,
            "sequence_length": 50,
            "batch_size": 2,
            "truncated_bptt_steps": 10,
        },
        "val": {
            "data_path": "./data/SUIT/images/",
            # 'auto' -> main.py resolves to the fold's labeled val with the
            # matching label family ({vendor}_fold{k}_val_{labels}.json).
            "annotations_path": "auto",
            "sequence_length": 50,
            "batch_size": 2,
            "truncated_bptt_steps": 10,
        },
        "num_workers": 12,
    }


def _emit(vendor: str, fold: int, labels: str, stage: str) -> Path:
    trainer = dict(TRAINER_BLOCK)
    trainer["ckpt_path"] = PRETRAIN_CKPT[labels]
    cfg = {
        "trainer": trainer,
        "model":   MODEL_BLOCK,
        "data":    _data_block(vendor, fold, labels, stage),
        "logging": LOGGING_BLOCK,
    }
    out = OUT_DIR / f"{vendor}_fold{fold}_{stage}_{labels}.yaml"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    written = []
    for vendor in ("ge", "mindray"):
        for fold in (0, 1, 2, 3):
            for labels in ("sam2", "ultrasam"):
                for stage in ("finetune", "ssl"):
                    written.append(_emit(vendor, fold, labels, stage))
    print(f"wrote {len(written)} configs to {OUT_DIR}/")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
