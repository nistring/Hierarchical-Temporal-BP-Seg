from lora_pytorch import LoRA
from lora_pytorch.modules.conv import (
    Conv1dLoRAModule,
    Conv2dLoRAModule,
    Conv3dLoRAModule,
    ConvType,
)
from typing import Type, Union, TypeVar, cast
import torch.nn as nn

ModuleType = TypeVar("ModuleType", bound=nn.Module)

class CustomLoRA(LoRA):
    """
    Custom extension of the LoRA class with a modified _from_conv implementation
    https://github.com/fkodom/lora-pytorch
    """
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    @classmethod
    def from_module(
        cls, module: ModuleType, rank: int, enabled: bool = True, is_root: bool = True
    ):
        if isinstance(module, nn.Linear):
            return LoRA._from_linear(module, rank)  # type: ignore
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if module.groups > 1:
                return module
            return LoRA._from_conv(module, rank)  # type: ignore
        elif isinstance(module, nn.Embedding):
            return LoRA._from_embedding(module, rank)
        elif isinstance(module, nn.MultiheadAttention):
            return LoRA._from_multihead_attention(module, rank)  # type: ignore

        for name, child in module.named_children():
            child = cast(ModuleType, child)
            module._modules[name] = cls.from_module(
                child, rank, enabled=enabled, is_root=False
            )

        if is_root:
            return LoRA(module, None, enabled=enabled)
        else:
            return module