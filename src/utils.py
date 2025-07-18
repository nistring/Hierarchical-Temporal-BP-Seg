import torch
from torchvision.transforms import v2

def process_video_stream(frame: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Process video stream and overlay predictions on the frame."""
    colors = torch.Tensor([
        (192, 255, 0), (0, 255, 192), (64, 0, 255), (255, 0, 64),
        (96, 255, 96), (255, 0, 255), (255, 128, 0), (0, 255, 0), (0, 128, 255)
    ]).to(masks.device)
    
    masks = masks / 2
    masks[0] += 0.5
    masks = v2.Resize(frame.shape[1:])(masks)
    
    frame = frame * masks[0] + (masks[1:, None] * colors[:masks.shape[0]-1, :, None, None]).sum(0)
    return frame.permute(1, 2, 0).cpu().numpy().astype("uint8")

def load_model(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    weights = torch.load(checkpoint_path)["state_dict"]
    weights = {k.replace("model.", "", 1): v for k, v in weights.items()}
    model.load_state_dict(weights)
    return model

def post_processing(masks: torch.Tensor) -> torch.Tensor:
    """Post-process segmentation masks."""
    masks = torch.nn.Softmax(dim=1)(masks[0])
    masks[:, 1:] = masks[:, 1:] * (masks[:, 0:1] < 0.5)
    
    mask_classes = masks.argmax(dim=1)
    masks[:, 2] = masks[:, 2] * torch.any(mask_classes == 1, dim=(1, 2), keepdim=True)
    masks[:, 4] = masks[:, 4] * torch.any(mask_classes == 3, dim=(1, 2), keepdim=True)
    masks[:, 6:] = masks[:, 6:] * torch.any(mask_classes == 4, dim=(1, 2), keepdim=True).unsqueeze(1)
    
    c56_dominant = ((mask_classes < 3) * (mask_classes > 0)).sum(dim=(1, 2), keepdim=True) > (mask_classes > 5).sum(dim=(1, 2), keepdim=True)
    masks[:, 1:3] = masks[:, 1:3] * c56_dominant[:, None]
    masks[:, 6:] = masks[:, 6:] * ~c56_dominant[:, None]
    masks[:, 0] = 1 - masks[:, 1:].sum(dim=1)
    
    return masks