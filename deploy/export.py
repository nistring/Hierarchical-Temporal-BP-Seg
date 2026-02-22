import argparse
from pathlib import Path

import torch
import yaml
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root
from src.model import TemporalSegmentationModel
from src.utils import replace_gelu_with_relu


HIDDEN = {
    "seq50": [  # SegformerB0
        (32, 104, 104),
        (64, 52, 52),
        (160, 26, 26),
        (256, 13, 13),
    ],
    "seq50_relu": [  # SegformerB0
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

# device-name -> (chipset, target_id)
# For qualcomm: target_id is the QAI Hub device name
# For mediatek: target_id is the SocModel enum name (MT69xx)
DEVICES = {
    "Galaxy Tab S8":  ("qualcomm", "Samsung Galaxy Tab S8"),
    "Galaxy Tab S9":  ("qualcomm", "Samsung Galaxy S23"),       # Snapdragon 8 Gen 2
    "Galaxy Tab S10": ("mediatek", "MT6989"),                    # Dimensity 9300+
    "Galaxy Tab S11": ("mediatek", "MT6991"),                    # Dimensity 9400+
}


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path:
        print(f"[export] No checkpoint found at {ckpt_path}, exporting randomly initialized weights.")
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
    print(f"[export] Loaded checkpoint. Missing: {len(missing)} Unexpected: {len(unexpected)}")


def build_model_from_config(config_path: Path, override_seq_len: int = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
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
    )

    if mcfg.get("use_relu", False):
        model = replace_gelu_with_relu(model)

    return model.eval()


def parse_args():
    p = argparse.ArgumentParser(description="Export temporal segmentation model for on-device deployment")
    p.add_argument("--config", type=Path, default="configs/seq50_relu.yaml", help="Path to training YAML config")
    p.add_argument("--height", type=int, default=416, help="Input frame height")
    p.add_argument("--width", type=int, default=416, help="Input frame width")
    p.add_argument("--channels", type=int, default=1, help="Input channels (1 for ultrasound)")
    p.add_argument(
        "--device-name", type=str, default="Galaxy Tab S9",
        choices=list(DEVICES.keys()),
        help="Target device",
    )
    return p.parse_args()


def export_mediatek(model, model_name, example_input, input_shape, soc, device_suffix):
    """Convert PyTorch model to TFLite with AOT compilation for MediaTek NPU."""
    import litert_torch
    from ai_edge_litert.aot.vendors.mediatek.target import SocModel, Target

    target = Target(soc_model=SocModel[soc])
    print(f"[mediatek] Converting + AOT compiling for {soc} with input shape: {input_shape}")
    compiled_models = (
        litert_torch
        .experimental_add_compilation_backend(target)
        .convert(model, tuple(example_input))
    )
    print(compiled_models.compilation_report())

    out_dir = f"deploy/{model_name}_{device_suffix}"
    compiled_models.export(out_dir, model_name=model_name)
    print(f"[mediatek] Exported AOT-compiled model to {out_dir}/")


def export_qualcomm(model, model_name, example_input, input_shape, hidden_state, hub_device, device_suffix):
    """Compile model via Qualcomm AI Hub."""
    import qai_hub as hub

    print(f"[qualcomm] Tracing model with input shape: {input_shape}")
    traced = torch.jit.trace(model, example_input)
    traced = torch.jit.freeze(traced)

    print("[qualcomm] Submitting compile job...")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(hub_device),
        input_specs=dict(
            image=input_shape,
            **{f"h{i}": hidden_state[i].shape for i in range(len(hidden_state))},
        ),
        options="--quantize_io --quantize_io_type uint8",
    )
    target_model = compile_job.get_target_model()
    print("[qualcomm] Compile complete.")

    print("[qualcomm] Submitting profile job...")
    hub.submit_profile_job(model=target_model, device=hub.Device(hub_device))
    print("[qualcomm] Profile submitted.")

    out_path = f"deploy/{model_name}_{device_suffix}.tflite"
    print("[qualcomm] Downloading compiled model...")
    target_model.download(out_path)
    print(f"[qualcomm] Saved compiled model to {out_path}")


def main():
    args = parse_args()
    chipset, target_id = DEVICES[args.device_name]
    device_suffix = args.device_name.split()[-1]  # e.g. "S10"

    model_name = args.config.stem
    model = build_model_from_config(args.config, override_seq_len=None)
    load_checkpoint_weights(model, f"lightning_logs/{model_name}/checkpoints/last.ckpt")

    input_shape = (1, args.channels, args.height, args.width)
    hidden_state = [torch.randn(2, 1, C, H, W) for C, H, W in HIDDEN[model_name]]
    example_input = [torch.randn(input_shape)] + hidden_state

    if chipset == "mediatek":
        export_mediatek(model, model_name, example_input, input_shape, target_id, device_suffix)
    else:
        export_qualcomm(model, model_name, example_input, input_shape, hidden_state, target_id, device_suffix)


if __name__ == "__main__":
    main()
