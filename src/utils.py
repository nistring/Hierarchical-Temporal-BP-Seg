import cv2
import torch
from torchvision.transforms import v2


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
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (0, 0, 255),
            (128, 255, 128),
            (128, 0, 128),
            (0, 128, 128),
            (0, 255, 0),
            (255, 128, 128),
            (128, 128, 255),
        ]
    ).to(masks.device)
    masks /= 2
    masks[0] += 0.5
    masks = v2.Resize(frame.shape[1:])(masks)
    frame = frame * masks[0] + (masks[1:, None] * colors[..., None, None]).sum(0)  # Overlay the masks on the frame
    frame = frame.permute(1, 2, 0).cpu().numpy().astype("uint8")
    return frame


def convert_to_coco_format(predictions: list) -> list:
    """
    Convert predictions to COCO format.

    Args:
        predictions (list): List of predictions.

    Returns:
        list: Predictions in COCO format.
    """
    coco_predictions = []
    for masks_pred, class_probs, image_id in predictions:
        for i, (mask, score) in enumerate(zip(masks_pred, class_probs)):
            if score > 0.5:
                contours, _ = cv2.findContours((mask > 0.5).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:  # COCO format requires at least 3 points (6 values)
                        segmentation.append(contour)
                if segmentation:
                    coco_pred = {"image_id": int(image_id), "category_id": i + 1, "segmentation": segmentation, "score": float(score)}
                    coco_predictions.append(coco_pred)
    return coco_predictions


def load_model(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load model weights from a checkpoint.

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
    return model


# This file is intentionally left blank.
