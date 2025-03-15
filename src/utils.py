import cv2
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
    frame = frame * masks[0] + (masks[1:, None] * colors[:, :, None, None]).sum(0)  # Overlay the masks on the frame
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
    model_weights = torch.load(checkpoint_path, weights_only=True)["state_dict"]
    # update keys by dropping `model.` only once
    for key in list(model_weights):
        new_key = key.replace("model.", "", 1)
        model_weights[new_key] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model = model.cuda()
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
    masks_suppressed = masks[:, 0] > 0.5
    masks[masks_suppressed.unsqueeze(1).repeat(1, masks.shape[1], 1, 1)] = 0
    masks[:, 0][masks_suppressed] = 1
    del masks_suppressed

    return masks
