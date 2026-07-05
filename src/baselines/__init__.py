"""
Lightweight medical-image segmentation baselines for R2.1-3 comparison.

Exposes a factory `make_baseline(name, num_classes, input_channels, image_size)`
that returns an nn.Module matching our trainer's (B, T, C, H, W) -> ((B, T, num_classes+1, H, W), ())
interface. Baselines are per-frame (no temporal state); the wrapper only
handles tensor reshaping and captures the pre-sigmoid logits via a forward
hook so we don't edit the vendored source.

Vendored (unmodified except for a stripped top-level `from utils import *` in
rolling_unet.py):
  - EGE-UNet (MICCAI 2023, 0.05 M params) — Apache-2.0
  - Rolling-UNet (AAAI 2024, 1.8 M params) — MIT
  - UltraLight VM-UNet (2024, 0.05 M params) — MIT (needs mamba_ssm)

The reviewer-named LightCF-Net, BFG&MSF-Net, CafeNet have no usable official
code. UltraLight VM-UNet stands in for the Mamba-based family (LightCF-Net).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .egeunet import EGEUNet
from .rolling_unet import Rolling_Unet_S, Rolling_Unet_M, Rolling_Unet_L
from .vanilla_unet import VanillaUNet
# `ultralight_vm_unet` imports `mamba_ssm`, whose prebuilt CUDA op can fail to
# load against a mismatched torch ABI. Defer that import to the call site so
# EGE-UNet / Rolling-UNet still work even when mamba_ssm is unavailable.


class _BaselineWrapper(nn.Module):
    """Match our trainer's expected signature.

    - Reshapes (B, T, C, H, W) -> (B*T, C, H, W) for the baseline's per-frame
      forward, and reshapes back.
    - Baselines finish with `torch.sigmoid(final_conv(x))`. For multi-class
      CrossEntropyLoss we need pre-sigmoid logits, so we register a forward
      hook on `self.final` (the last Conv2d) to capture its output. This
      keeps the vendored source files unmodified.
    - Returns ((B, T, C_out, H, W), ()) — empty tuple for hidden state.
    """

    def __init__(self, baseline: nn.Module, sigmoid_wrapped: bool = True,
                 num_classes: int = 0, deep_supervision: bool = False):
        super().__init__()
        self.baseline = baseline
        self.sigmoid_wrapped = sigmoid_wrapped
        # `num_classes` excludes background — matches TemporalSegmentationModel
        # so SegmentationTrainer.test_step's `self.model.num_classes` works.
        self.num_classes = num_classes
        # Deep supervision: EGE-UNet's auxiliary `gt_pre1..gt_pre5` heads
        # (1-channel each, the paper's binary-segmentation aux signal).
        # When True, SegmentationTrainer.compute_loss reads `aux_sigmoids`
        # and applies BCE against the binary foreground mask.
        self.deep_supervision = deep_supervision
        self._captured_logits: Optional[torch.Tensor] = None
        self.aux_sigmoids: Optional[Tuple[torch.Tensor, ...]] = None

        if sigmoid_wrapped:
            # EGE-UNet / UltraLight / Rolling-Unet all expose `self.final`
            # as the last Conv2d -> (B, num_classes, H/2, W/2) pre-upsample
            # or similar. We capture its output, then re-do the upsample in
            # the wrapper's forward to get full-resolution logits.
            assert hasattr(baseline, "final"), f"{type(baseline).__name__} has no `final`"
            baseline.final.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        self._captured_logits = output

    def forward(self, x: torch.Tensor, hidden_state=None) -> Tuple[torch.Tensor, tuple]:
        squeezed = False
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1
            x_flat = x
            squeezed = True

        sig_out = self.baseline(x_flat)  # returns sigmoid(logits)
        # EGE-UNet with gt_ds=True returns ((aux1..aux5), final); other
        # models return a single tensor. Separate the two cases.
        if (self.deep_supervision and isinstance(sig_out, tuple)
                and len(sig_out) == 2 and isinstance(sig_out[0], tuple)):
            self.aux_sigmoids = sig_out[0]
            sig_out = sig_out[1]
        else:
            self.aux_sigmoids = None
            if isinstance(sig_out, tuple):
                sig_out = sig_out[-1]

        if self.sigmoid_wrapped and self._captured_logits is not None:
            logits = self._captured_logits
            # `final` output may be half-resolution (EGE-UNet / UltraLight
            # apply a final bilinear upsample by 2 AFTER sigmoid). Upsample
            # our captured logits to the sigmoid output's spatial size.
            if logits.shape[-2:] != sig_out.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=sig_out.shape[-2:], mode="bilinear", align_corners=True)
            out = logits
            self._captured_logits = None
        else:
            # Rolling-UNet has no sigmoid — sig_out is already logits.
            out = sig_out

        if not squeezed:
            out = out.reshape(B, T, *out.shape[1:])
        return out, ()


def make_baseline(name: str, num_classes: int, input_channels: int = 1,
                  image_size: Tuple[int, int] = (416, 416)) -> nn.Module:
    """Return a baseline wrapped for our trainer.

    num_classes: total classes including background (e.g., 1+8=9)."""
    h, w = image_size
    assert h == w, f"Baselines assume square input; got {image_size}"
    name = name.lower()

    if name == "ege_unet":
        # gt_ds=True is required: the `else` branch of the reference forward
        # calls group_aggregation_bridge without its mandatory `mask` arg.
        # With gt_ds=True the forward returns (aux_tuple, final); the wrapper
        # exposes the aux outputs via `aux_sigmoids` and the trainer applies
        # binary BCE on them against the foreground mask (the original paper's
        # multi-scale auxiliary supervision, adapted to multi-class).
        m = EGEUNet(num_classes=num_classes, input_channels=input_channels,
                    c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True)
        return _BaselineWrapper(m, sigmoid_wrapped=True,
                                num_classes=num_classes - 1, deep_supervision=True)

    if name == "ultralight_vm_unet":
        from .ultralight_vm_unet import UltraLight_VM_UNet
        m = UltraLight_VM_UNet(num_classes=num_classes, input_channels=input_channels,
                               c_list=[8, 16, 24, 32, 48, 64], split_att="fc", bridge=True)
        return _BaselineWrapper(m, sigmoid_wrapped=True, num_classes=num_classes - 1)

    if name in ("rolling_unet_s", "rolling_unet"):
        m = Rolling_Unet_S(num_classes=num_classes, input_channels=input_channels,
                           img_size=h, deep_supervision=False)
        return _BaselineWrapper(m, sigmoid_wrapped=False, num_classes=num_classes - 1)

    if name == "rolling_unet_m":
        m = Rolling_Unet_M(num_classes=num_classes, input_channels=input_channels,
                           img_size=h, deep_supervision=False)
        return _BaselineWrapper(m, sigmoid_wrapped=False, num_classes=num_classes - 1)

    if name == "rolling_unet_l":
        m = Rolling_Unet_L(num_classes=num_classes, input_channels=input_channels,
                           img_size=h, deep_supervision=False)
        return _BaselineWrapper(m, sigmoid_wrapped=False, num_classes=num_classes - 1)

    if name == "vanilla_unet":
        # Original Ronneberger 2015 UNet, no pretraining, no sigmoid wrapping
        # (final 1x1 conv emits raw logits). Channel schedule 64..1024.
        m = VanillaUNet(num_classes=num_classes, input_channels=input_channels)
        return _BaselineWrapper(m, sigmoid_wrapped=False, num_classes=num_classes - 1)

    raise ValueError(f"Unknown baseline: {name}")
