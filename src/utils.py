import torch
from torchvision.transforms import v2

def process_video_stream(frame: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Process video stream and overlay predictions on the frame."""
    colors = torch.Tensor([
        (192, 255, 0), (0, 255, 192), (64, 0, 255), (255, 0, 64),
        (96, 255, 96), (0, 255, 0), (255, 128, 0), (255, 0, 255),
    ], device=masks.device)
    
    masks = masks / 2
    
    # masks = v2.Resize(frame.shape[1:])(masks)
    frame = frame * (masks[0:1] + 0.5) + (masks[1:].unsqueeze(1) * colors.unsqueeze(-1).unsqueeze(-1)).sum(0)
    return frame.permute(1, 2, 0) # for exporting to TFLite
    # return frame.permute(1, 2, 0).cpu().numpy().astype("uint8")

def load_model(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    weights = torch.load(checkpoint_path)["state_dict"]
    weights = {k.replace("model.", "", 1): v for k, v in weights.items()}
    model.load_state_dict(weights)
    return model

def post_processing(masks: torch.Tensor) -> torch.Tensor:
    masks = torch.nn.Softmax(dim=1)(masks)

    other_classes = masks[:, 1:] * (masks[:, 0:1] < 0.5)
    class0 = 1 - other_classes.sum(dim=1, keepdim=True)
    masks = torch.cat([class0, other_classes], dim=1)
    return masks

def replace_gelu_with_relu(model: torch.nn.Module) -> torch.nn.Module:
    """Replace all GeLU activations with ReLU in a model."""
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.ReLU())
            elif isinstance(child, torch.nn.Module):
                replace_gelu_with_relu(child)
    return model
