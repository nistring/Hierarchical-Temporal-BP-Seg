import torch
from torchvision.transforms import v2
import numpy as np

def process_video_stream(frame: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Process video stream and overlay predictions on the frame.

    Args:
        frame (torch.Tensor): Input video frame.
        masks (torch.Tensor): Predicted masks.

    Returns:
        torch.Tensor: Processed frame with overlaid predictions.
    """
    colors = torch.Tensor(
        [
            (192, 255, 0),  # C5
            (0, 255, 192),  # C6
            (64, 0, 255),  # C7
            (255, 0, 64),  # C8/LT
            (96, 255, 96),  # UT
            (255, 0, 255),  # MT
            (255, 128, 0),  # SSN
            (0, 255, 0),  # AD
            (0, 128, 255),  # PD
        ]
    ).to(masks.device)
    masks /= 2
    masks[0] += 0.5
    masks = v2.Resize(frame.shape[1:])(masks)
    # masks[1:] = torch.zeros_like(masks[1:]).scatter_(0, masks[1:].argmax(dim=0, keepdim=True), 1 - masks[0:1])
    frame = frame * masks[0] + (masks[1:, None] * colors[:masks.shape[0]-1, :, None, None]).sum(0)  # Overlay the masks on the frame
    frame = frame.permute(1, 2, 0).cpu().numpy().astype("uint8")
    return frame


def load_model(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load model weights from a checkpoint to a specific device.

    Args:
        model (torch.nn.Module): Model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    model_weights = torch.load(checkpoint_path)["state_dict"]
    # update keys by dropping `model.` only once
    for key in list(model_weights):
        new_key = key.replace("model.", "", 1)
        model_weights[new_key] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    return model


def post_processing(masks: torch.Tensor) -> torch.Tensor:
    """
    Post-process the segmentation masks.
    Used only in the test and demo scripts.

    Args:
        masks (torch.Tensor): Segmentation masks.

    Returns:
        torch.Tensor: Processed segmentation masks.
    """
    masks = torch.nn.Softmax(dim=1)(masks[0])  # drop batch_size dimension
    masks[:, 1:] = masks[:, 1:] * (masks[:, 0:1] < 0.5)  # Suppress background

    masks_by_class = masks.argmax(dim=1) # Get the class with the highest probability for each pixel
    masks[:, 2] = masks[:, 2] * torch.any(masks_by_class == 1, dim=(1, 2), keepdim=True)  # C6 cannot exist without C5
    masks[:, 4] = masks[:, 4] * torch.any(masks_by_class == 3, dim=(1, 2), keepdim=True)  # C8 cannot exist without C7
    masks[:, 6:] = masks[:, 6:] * torch.any(masks_by_class == 4, dim=(1, 2), keepdim=True).unsqueeze(1)   # SSN, AD, PD cannot exist without LT
    C56_ADPDSSN = ((masks_by_class < 3) * (masks_by_class > 0)).sum(dim=(1, 2), keepdim=True) > (masks_by_class > 5).sum(dim=(1, 2), keepdim=True)  # C5 + C6 > AD + PD + SSN
    masks[:, 1:3] = masks[:, 1:3] * C56_ADPDSSN[:, None]  # C5 and C6 cannot exist with AD, PD, SSN
    masks[:, 6:] = masks[:, 6:] * ~C56_ADPDSSN[:, None]  # AD, PD, SSN cannot exist with C5 or C6

    masks[:, 0] = 1 - masks[:, 1:].sum(dim=1)  # Background is the rest

    return masks
