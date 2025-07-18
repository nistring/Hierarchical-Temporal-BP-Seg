import torch
import torch.nn as nn
import torch.nn.functional as F

class SparsityLoss(nn.Module):
    """Encourages spatially compact predictions by penalizing scattered activations."""
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions):
        """Args: predictions: (B * T, C, H, W) logits"""
        probs = F.softmax(predictions, dim=1)[:, 1:]  # Exclude background
        h, w = probs.shape[2], probs.shape[3]
        
        # Normalized coordinates [-1, 1]
        h_coords = torch.arange(h, device=probs.device) / (h - 1) * 2 - 1
        w_coords = torch.arange(w, device=probs.device) / (w - 1) * 2 - 1
        
        # Center of mass
        total_mass = probs.sum(dim=(2,3), keepdim=True) + 1e-8
        center_h = (probs * h_coords.view(-1, 1)).sum(dim=(2,3), keepdim=True) / total_mass
        center_w = (probs * w_coords.view(1, -1)).sum(dim=(2,3), keepdim=True) / total_mass
        
        # Variance around center
        h_var = (probs * (h_coords.view(1, 1, -1, 1) - center_h) ** 2).sum(dim=(2,3)) / total_mass.squeeze()
        w_var = (probs * (w_coords.view(1, 1, 1, -1) - center_w) ** 2).sum(dim=(2,3)) / total_mass.squeeze()
        
        return (h_var + w_var).mean()

class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video segmentation."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred):
        """
        Args: pred (torch.Tensor): Shape (B, T, C, H, W)
        Returns: torch.Tensor: Temporal consistency loss
        """
        if pred.dim() != 5 or pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)
        
        pred_probs = F.softmax(pred, dim=2)
        return torch.norm(pred_probs[:, 1:] - pred_probs[:, :-1], p=2, dim=2).mean()

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