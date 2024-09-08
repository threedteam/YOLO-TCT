import torch
import torch.nn as nn

from ..modules.conv import Conv
from ..modules.block import RepNCSPELAN4

from .PSA import PSA, HPSA, CPSA, HCPSA

"""
RepNCSPELAN4: RepNCSPELAN4, original.
ResELAN4: ML+RepNCSPELAN4
PSA_LRRELAN4: ML+RepNCSPELAN4+PSA
HPSA_LRRELAN4: ML+RepNCSPELAN4+HPSA
CPSA_LRRELAN4: ML+RepNCSPELAN4+CPSA
HCPSA_LRRELAN4: ML+RepNCSPELAN4+HCPSA
"""

class ResELAN4(RepNCSPELAN4):
    """Residual ELAN block with CSPELAN4."""
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__(c1, c2, c3, c4, n)
        # short cut
        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=1, act=False)) if c1 != c2 else nn.Identity()
        
        # REFINEMENT place holder
        self.refinement = nn.Identity()

    
    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.refinement(self.cv4(torch.cat(y, 1)))+self.shortcut(x)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.refinement(self.cv4(torch.cat(y, 1)))+self.shortcut(x) 

class PSA_LRRELAN4(ResELAN4):
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__(c1, c2, c3, c4, n)
        self.refinement = PSA(c2)

class HPSA_LRRELAN4(ResELAN4):
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__(c1, c2, c3, c4, n) 
        self.refinement = HPSA(c2)

class CPSA_LRRELAN4(ResELAN4):
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__(c1, c2, c3, c4, n) 
        self.refinement = CPSA(c2)

class HCPSA_LRRELAN4(ResELAN4):
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__(c1, c2, c3, c4, n) 
        self.refinement = HCPSA(c2)