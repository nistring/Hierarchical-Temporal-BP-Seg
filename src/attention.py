from torch import nn
import torch

class ECANet(nn.Module):
    """
    Efficient Channel Attention (ECA) Module
    Paper: https://arxiv.org/abs/1910.03151
    """
    def __init__(self, kernel_size=3):
        super(ECANet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SimAM(nn.Module):
    """
    Simple, Parameter-Free Attention Module (SimAM)
    Paper: https://arxiv.org/abs/2101.11297
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda
        
    def forward(self, x):
        # Spatial attention
        b, c, h, w = x.size()
        
        # Calculate the parameters required by the spatial attention
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        x_norm = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # Apply the spatial attention (element-wise multiplication)
        return x * torch.sigmoid(x_norm)