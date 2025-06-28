import torch
import torch.nn as nn
import torch.nn.functional as F

class SparsityLoss(nn.Module):
    """
    Encourages spatially compact predictions by penalizing scattered activations.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions):
        """
        Args:
            predictions: (B * T, C, H, W) logits
        """
        probs = F.softmax(predictions, dim=1)[:, 1:]  # Exclude background class (assumed to be class 0)

        # Create coordinate grids
        h_coords = torch.arange(probs.shape[2], device=probs.device) / (probs.shape[2] - 1) * 2 - 1  # Normalize to [-1, 1]
        w_coords = torch.arange(probs.shape[3], device=probs.device) / (probs.shape[3] - 1) * 2 - 1  # Normalize to [-1, 1]
        
        # Compute center of mass
        total_mass = torch.sum(probs, dim=(2,3), keepdim=True) + 1e-8
        center_h = torch.sum(probs * h_coords.reshape(-1, 1), dim=(2,3), keepdim=True) / total_mass
        center_w = torch.sum(probs * w_coords.reshape(1, -1), dim=(2,3), keepdim=True) / total_mass
        
        # Compute variance (spread) around center of mass
        h_var = torch.sum(probs * (h_coords.reshape(1, 1, -1, 1) - center_h) ** 2, dim=(2,3), keepdim=True) / total_mass
        w_var = torch.sum(probs * (w_coords.reshape(1, 1, 1, -1) - center_w) ** 2, dim=(2,3), keepdim=True) / total_mass
    
        return (h_var + w_var).mean()

class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss that enforces consistent predictions across frames.
    
    References:
    - Nilsson, D., & Sminchisescu, C. (2018). Semantic video segmentation by gated recurrent
      flow propagation. In Proceedings of the IEEE Conference on Computer Vision and Pattern
      Recognition (CVPR).
    - Lei, P., & Todorovic, S. (2018). Temporal deformable residual networks for
      action segmentation in videos. In Proceedings of the IEEE Conference on Computer
      Vision and Pattern Recognition (CVPR).
    - Liu, Y., et al. (2019). Efficient Semantic Video Segmentation with Per-frame
      Inference. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).
    
    Code References:
    - https://github.com/tensorflow/models/blob/master/research/vid2depth/consistency_losses.py
    - https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/losses.py
    - https://github.com/shelhamer/clockwork-fcn/blob/master/temporal_modules.py
    """
    def __init__(self):
        """
        Initialize the temporal consistency loss.
        """
        super().__init__()

        
    def forward(self, pred):
        """
        Compute the temporal consistency loss between adjacent frames.
        
        Args:
            pred (torch.Tensor): Predicted segmentation of shape (B, T, C, H, W),
                                 where T is the sequence length.
                                 
        Returns:
            torch.Tensor: Temporal consistency loss value.
        """
        if pred.dim() != 5:
            raise ValueError("Expected 5D tensor for temporal predictions")
        if pred.shape[1] <= 1:
            # No temporal loss for single frame
            return torch.tensor(0.0, device=pred.device)
        
        # Calculate difference between consecutive frames
        # Using softmax to get probabilities
        pred_probs = F.softmax(pred, dim=2)
        temporal_diff = torch.norm(pred_probs[:, 1:] - pred_probs[:, :-1], p=2, dim=2).mean()
        
        return temporal_diff
