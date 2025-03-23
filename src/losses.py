import torch
import torch.nn as nn
import torch.nn.functional as F

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
