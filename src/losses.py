import torch
import torch.nn as nn
import torch.nn.functional as F

class ExclusionLoss(nn.Module):
    """Exclusion Loss for video segmentation.

    Encourages mutual exclusivity between predefined groups of classes.
    Groups are specified over foreground class indices [0..num_classes-1],
    where index 0 corresponds to class id 1 in the dataset if background is 0.
    If groups is None, falls back to legacy fixed grouping.
    """
    def __init__(self, groups: list[list[int]] | None = None):
        super().__init__()
        self.groups = groups
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B*T, C, H, W) segmentation logits, C includes background at channel 0
        Returns:
            Scalar loss: sum over pairwise group overlaps
        """
        probs = F.softmax(predictions, dim=1)[:, 1:]  # Exclude background -> (B*T, C_fg, H, W)

        # Legacy fallback: assume [root (0-1), trunk (4), division (5+)] over foreground channels
        if not self.groups:
            root = probs[:, :2].sum(1).mean(dim=(1, 2))
            trunk = probs[:, 4:5].sum(1).mean(dim=(1, 2))
            division = probs[:, 5:].sum(1).mean(dim=(1, 2))
            return (root * trunk + trunk * division + root * division).mean()

        # Compute mean activation per group per sample
        group_means = []
        for idxs in self.groups:
            if len(idxs) == 0:
                continue
            g_prob = probs[:, idxs].sum(1)  # (B*T, H, W)
            group_means.append(g_prob.mean(dim=(1, 2)))  # (B*T,)

        if len(group_means) <= 1:
            return predictions.new_tensor(0.0)

        # Pairwise products of group means -> encourage exclusivity
        loss = 0.0
        n = len(group_means)
        for i in range(n):
            for j in range(i + 1, n):
                loss = loss + group_means[i] * group_means[j]
        return loss.mean()
        

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
