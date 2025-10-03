import argparse
import yaml
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root
from src.model import TemporalSegmentationModel  # noqa: E402


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path or not ckpt_path.exists():
        print(f"[export] No checkpoint found at {ckpt_path}, exporting randomly initialized weights.")
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
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
    p = argparse.ArgumentParser(description="Export temporal segmentation model to ExecuTorch (XNNPACK)")
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to YAML config used during training")
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional path to model checkpoint (.ckpt)")
    p.add_argument("--seq-len", type=int, default=10, help="Override sequence length for export (default: from config or 4)")
    p.add_argument("--height", type=int, default=416, help="Input frame height")
    p.add_argument("--width", type=int, default=416, help="Input frame width")
    p.add_argument("--output", type=Path, default=Path("temporal_seg_xnnpack_fp32.pte"), help="Output .pte filename")
    p.add_argument("--channels", type=int, default=1, help="Number of input channels (default 1 for ultrasound)")
    p.add_argument("--device", type=str, default="cpu", help="Device for export graph tracing (keep cpu for portability)")
    p.add_argument("--backend", choices=["vulkan", "xnnpack"], default="vulkan", help="Execution backend partitioner to target")
    return p.parse_args()

def main():
    args = parse_args()
    model, seq_len, num_classes = build_model_from_config(args.config, args.seq_len)
    if args.checkpoint:
        load_checkpoint_weights(model, args.checkpoint)
    model = model.to(args.device)

    # Create sample input
    sample_video = torch.randn(1, seq_len, args.channels, args.height, args.width, device=args.device, dtype=torch.float32)

    # Run one dry forward to obtain an initialized hidden state structure with concrete tensor shapes
    with torch.no_grad():
        _, sample_hidden_state = model(sample_video, None)

    # torch.export requires inputs as a tuple. We include both input video and hidden state.
    sample_inputs = (sample_video, sample_hidden_state)
    print("[export] Sample video shape:", tuple(sample_video.shape))
    print("[export] Hidden state structure serialized for runtime re-creation.")

    exported_program = torch.export.export(model, sample_inputs)

    backend = args.backend

    partitioner_obj = XnnpackPartitioner() if backend == "xnnpack" else VulkanPartitioner()

    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner_obj],
    ).to_executorch()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        et_program.write_to_file(f)


if __name__ == "__main__":
    main()