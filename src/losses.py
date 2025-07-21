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

class PerceptualConsistencyLoss(nn.Module):
    """Perceptual Consistency Loss for video segmentation."""
    """https://arxiv.org/pdf/2110.12385"""
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, feature):
        """
        Args:
            predictions: (B, T, C, H, W) segmentation logits
            feature: (B, T, C_feat, H_feat, W_feat) feature tensor
        """
        predictions = F.softmax(predictions, dim=2)

        loss = 0
        C_feat, H_feat, W_feat = feature.shape[2:]
        hw = H_feat * W_feat

        # Normalize and reshape features
        feat_a = F.normalize(feature[:, :-1].reshape(-1, C_feat, hw), dim=1)
        feat_b = F.normalize(feature[:, 1:].reshape(-1, C_feat, hw), dim=1)

        seg_a = predictions[:, :-1].reshape(-1, *predictions.shape[2:])
        seg_b = predictions[:, 1:].reshape(-1, *predictions.shape[2:])

        # Resize predictions if needed
        if seg_a.shape[-2:] != (H_feat, W_feat):
            seg_a = F.interpolate(seg_a, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            seg_b = F.interpolate(seg_b, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        
        seg_a = seg_a.reshape(-1, seg_a.shape[1], hw)
        seg_b = seg_b.reshape(-1, seg_b.shape[1], hw)

        # for bt in range(feat_a.shape[0]):
        #     # Compute correlations
        #     corr = torch.matmul(feat_a[bt].T, feat_b[bt])
        #     seg_corr = torch.matmul(seg_a[bt].T, seg_b[bt])
            
        #     # Consistency metrics for both dimensions
        #     def consistency(corr, seg_corr, dim):
        #         max_unconstrained = torch.max(corr, dim=dim)[0]
        #         max_constrained = torch.max(corr * seg_corr, dim=dim)[0]
        #         mean_corr = torch.mean(corr, dim=dim)
        #         return (max_constrained - mean_corr) / (max_unconstrained - mean_corr)
            
        #     loss += torch.min(consistency(corr, seg_corr, 1), consistency(corr, seg_corr, 0)).mean()

        # Compute correlations
        corr = torch.bmm(feat_a.transpose(1, 2), feat_b)
        seg_corr = torch.bmm(seg_a.transpose(1, 2), seg_b)
        
        # Consistency metrics for both dimensions
        def consistency(corr, seg_corr, dim):
            max_unconstrained = torch.max(corr, dim=dim)[0]
            max_constrained = torch.max(corr * seg_corr, dim=dim)[0]
            mean_corr = torch.mean(corr, dim=dim)
            return (max_constrained - mean_corr) / (max_unconstrained - mean_corr)
        
        loss = torch.min(consistency(corr, seg_corr, 2), consistency(corr, seg_corr, 1)).mean()

        return 1 - loss
