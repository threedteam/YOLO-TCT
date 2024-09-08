import torch
import torch.nn as nn

from ..modules.conv import Conv
from .PSA import HCPSA

class LRRSPPELAN(nn.Module):
    """LRR-SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)
        
        # short cut
        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=1, act=False)) if c1 != c2 else nn.Identity()
        # REFINEMENT
        self.refinement = HCPSA(c2)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.refinement(self.cv5(torch.cat(y, 1)))+self.shortcut(x)