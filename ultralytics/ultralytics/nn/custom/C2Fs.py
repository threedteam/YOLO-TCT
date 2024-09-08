import torch
import torch.nn as nn

from ..modules.conv import Conv
from ..modules.block import C2f

from .PSA import PSA, HPSA, CPSA, HCPSA

class ResC2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # short cut
        self.shortcut = nn.Sequential(Conv(c1, c2, k=1, s=1, act=False)) if c1 != c2 else nn.Identity()
        
        # REFINEMENT place holder
        self.refinement = nn.Identity()
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.refinement(self.cv2(torch.cat(y, 1)))+self.shortcut(x)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.refinement(self.cv2(torch.cat(y, 1)))+self.shortcut(x)

class NoResC2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # REFINEMENT place holder
        self.refinement = nn.Identity()
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.refinement(self.cv2(torch.cat(y, 1)))
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.refinement(self.cv2(torch.cat(y, 1)))

class HCPSA_C2f(ResC2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) 
        self.refinement = HCPSA(c2)

class CPSA_C2f(ResC2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) 
        self.refinement = CPSA(c2)
        
class HCPSA_noRes_C2f(ResC2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) 
        self.refinement = HCPSA(c2)

class CPSA_noRes_C2f(ResC2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) 
        self.refinement = CPSA(c2)