import argparse
from pathlib import Path

import qai_hub as hub
import torch
import numpy as np

import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root
from src.model import TemporalSegmentationModel, TemporalSegmentationExportWrapper


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path or not ckpt_path.exists():
        print(f"[export] No checkpoint found at {ckpt_path}, exporting randomly initialized weights.")
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip potential Lightning 'model.' prefix
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[export] Loaded checkpoint. Missing: {len(missing)} Unexpected: {len(unexpected)}")


def build_model_from_config(config_path: Path, override_seq_len: int = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    # Provide sane mobile-friendly defaults if not present
    encoder_name = mcfg.get("encoder_name", "mit_b0")
    segmentation_model_name = mcfg.get("segmentation_model_name", "Segformer")
    num_classes = mcfg.get("num_classes", 8)
    temporal_model = mcfg.get("temporal_model", "ConvGRU")
    num_layers = mcfg.get("num_layers", 1)
    kernel_size = tuple(mcfg.get("kernel_size", [3, 3]))
    dilation = mcfg.get("dilation", 1)
    encoder_depth = mcfg.get("encoder_depth", 5)
    temporal_depth = mcfg.get("temporal_depth", 1)
    conv_type = mcfg.get("conv_type", "standard")
    model_kwargs = mcfg.get("model_kwargs", {})

    model = TemporalSegmentationModel(
        encoder_name=encoder_name,
        segmentation_model_name=segmentation_model_name,
        num_classes=num_classes,
        temporal_model=temporal_model,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dilation=dilation,
        encoder_depth=encoder_depth,
        temporal_depth=temporal_depth,
        conv_type=conv_type,
        **model_kwargs,
    ).eval()

    seq_len = override_seq_len or cfg["data"]["train"].get("sequence_length", 4)
    return model, seq_len, num_classes

def parse_args():
    p = argparse.ArgumentParser(description="Compile custom temporal segmentation model to Qualcomm AI Hub")
    p.add_argument("--config", type=Path, default="configs/deeplab.yaml", help="Path to training YAML config")
    p.add_argument("--checkpoint", default="lightning_logs/deeplab/checkpoints/last.ckpt", type=Path, help="Optional .ckpt path")
    p.add_argument("--height", type=int, default=416, help="Input frame height")
    p.add_argument("--width", type=int, default=416, help="Input frame width")
    p.add_argument("--channels", type=int, default=1, help="Input channels (1 for ultrasound)")
    p.add_argument("--device-name", type=str, default="Samsung Galaxy S23", help="Target device name on QAI Hub")
    p.add_argument("--download-name", type=str, default="deeplab.tflite", help="Filename for downloaded model")
    p.add_argument("--run-sample", action="store_true", help="Run a sample on-device inference after compile")
    return p.parse_args()


def main():
    args = parse_args()

    # Build core model (sequence length not needed for single-frame wrapper)
    model, _, num_classes = build_model_from_config(args.config, override_seq_len=None)
    if args.checkpoint:
        load_checkpoint_weights(model, args.checkpoint)
    # Wrap for flat tensor list output (out + hidden states)
    model = model.eval()
    # Create example input (single frame)
    input_shape = (1, 1, args.channels, args.height, args.width)
    example_input = torch.randn(input_shape)
    # Build hidden_state tensors (each: [2, 1, C, H, W] => stacked (c, h))
    hidden_state = model(example_input, None)[1]
    print(f"[export] Example input shape: {example_input.shape}, hidden state shapes: {[h[0].shape for h in hidden_state]}")


if __name__ == "__main__":
    main()