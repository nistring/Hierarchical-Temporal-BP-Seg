import argparse
from pathlib import Path

import qai_hub as hub
import torch
import numpy as np

import yaml
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root
from src.model import TemporalSegmentationModel


HIDDEN = {
    "seq50": [  # SegformerB0
        (32, 104, 104),
        (64, 52, 52),
        (160, 26, 26),
        (256, 13, 13),
    ],
    "deeplab": [  # deeplabv3+efficientnetb0, with temporal depth of 1
        (32, 208, 208),
        (24, 104, 104),
        (40, 52, 52),
        (112, 26, 26),
        (320, 26, 26),
    ],
    "unet": [  # UNet with resnet18 backbone
        (64, 208, 208),
        (64, 104, 104),
        (128, 52, 52),
        (256, 26, 26),
        (512, 13, 13),
    ],
}


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path:
        print(
            f"[export] No checkpoint found at {ckpt_path}, exporting randomly initialized weights."
        )
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip potential Lightning 'model.' prefix
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state[k[len("model.") :]] = v
        else:
            new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(
        f"[export] Loaded checkpoint. Missing: {len(missing)} Unexpected: {len(unexpected)}"
    )


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

    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Compile custom temporal segmentation model to Qualcomm AI Hub"
    )
    p.add_argument(
        "--config",
        type=Path,
        default="configs/unet.yaml",
        help="Path to training YAML config",
    )
    p.add_argument("--height", type=int, default=416, help="Input frame height")
    p.add_argument("--width", type=int, default=416, help="Input frame width")
    p.add_argument(
        "--channels", type=int, default=1, help="Input channels (1 for ultrasound)"
    )
    p.add_argument(
        "--device-name",
        type=str,
        default="Samsung Galaxy S23",
        help="Target device name on QAI Hub",
    )
    p.add_argument(
        "--run-sample",
        action="store_true",
        help="Run a sample on-device inference after compile",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build core model (sequence length not needed for single-frame wrapper)
    model_name = args.config.stem
    model = build_model_from_config(args.config, override_seq_len=None)
    load_checkpoint_weights(model, f"lightning_logs/{model_name}/checkpoints/last.ckpt")
    # Create example input (single frame)
    input_shape = (1, args.channels, args.height, args.width)
    example_input = torch.randn(input_shape)

    # Build hidden_state tensors
    hidden_specs = [  # deeplabv3+efficientnetb0, with temporal depth of 1
        (320, 26, 26),
    ]

    hidden_specs = HIDDEN[model_name]

    hidden_state = []
    for C, H, W in hidden_specs:
        h = torch.randn(2, 1, C, H, W)
        hidden_state.append(h)

    example_input = [example_input] + hidden_state
    # TorchScript trace
    print("[qualcomm] Tracing model with input shape:", input_shape)
    traced = torch.jit.trace(model, example_input)
    traced = torch.jit.freeze(traced)

    # Submit compile job
    print("[qualcomm] Submitting compile job...")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(args.device_name),
        input_specs=dict(
            image=input_shape,
            h0=hidden_state[0].shape,
            h1=hidden_state[1].shape,
            h2=hidden_state[2].shape,
            h3=hidden_state[3].shape,
            h4=hidden_state[4].shape,
        ),  # list of shapes
    )
    target_model = compile_job.get_target_model()
    print("[qualcomm] Compile complete.")

    # Optional profiling
    print("[qualcomm] Submitting profile job...")
    hub.submit_profile_job(
        model=target_model,
        device=hub.Device(args.device_name),
    )
    print("[qualcomm] Profile submitted.")

    # # Download compiled artifact
    print("[qualcomm] Downloading compiled model...")
    target_model.download(f"deploy/{model_name}.tflite")
    print(f"[qualcomm] Saved compiled model to {model_name}.tflite")

    # Run inference using the on-device model on the input image
    # inference_job = hub.submit_inference_job(
    #     model=target_model,
    #     device=hub.Device(args.device_name),
    #     inputs=dict(image=[example_input[0].numpy().astype(np.float32)], h0=[hidden_state[0].numpy().astype(np.float32)]),
    # )


if __name__ == "__main__":
    main()
