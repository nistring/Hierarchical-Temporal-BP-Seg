import torch
import torch.nn as nn
import torch.nn.functional as F

class ExclusionLoss(nn.Module):
    """Exclusion Loss for video segmentation."""
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions):
        """
        Args:
            predictions: (B * T, C, H, W) segmentation logits
        """
        probs = F.softmax(predictions, dim=1)[:, 1:]  # Exclude background
        root = probs[:, :2].sum(1).mean(dim=(1, 2))
        trunk = probs[:, 4].mean(dim=(1, 2))
        division = probs[:, 5:].sum(1).mean(dim=(1, 2))

        return (root * trunk + trunk * division + root * division).mean()
        

class ContrastiveLoss(nn.Module):
    """Encourages spatially compact predictions by penalizing scattered activations."""
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions):
        """Args: predictions: (B * T, C, H, W) logits"""
        probs = F.softmax(predictions, dim=1)[:, 1:]  # Exclude background
        b, num_classes, h, w = probs.shape
        
        # Normalized coordinates [-1, 1]
        h_coords = torch.arange(h, device=probs.device) / (h - 1) * 2 - 1
        w_coords = torch.arange(w, device=probs.device) / (w - 1) * 2 - 1
        
        # Center of mass for each class
        total_mass = probs.sum(dim=(2,3), keepdim=True) + 1e-8
        center_h = (probs * h_coords.view(1, 1, -1, 1)).sum(dim=(2,3), keepdim=True) / total_mass
        center_w = (probs * w_coords.view(1, 1, 1, -1)).sum(dim=(2,3), keepdim=True) / total_mass

        coords = (h_coords.view(1, 1, -1, 1) - center_h) ** 2 + (w_coords.view(1, 1, 1, -1) - center_w) ** 2  # (B * T, C, H, W)

        positive = (probs**2 * coords).mean()
        coords = coords * (total_mass / h / w)
        negative = 0
        for i in range(num_classes):
            non_i = torch.arange(num_classes) != i
            negative -= (probs[:, i:i+1] * coords[:, non_i]).mean()
        return positive, negative

class TemporalConsistencyLoss(nn.Module):
    """Temporal Consistency Loss for video segmentation."""
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions):
        predictions = F.softmax(predictions, dim=2)

        # Compute pairwise differences
        return (predictions[:, :-1] - predictions[:, 1:]).abs().mean()
