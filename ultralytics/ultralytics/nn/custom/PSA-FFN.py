import torch
import torch.nn as nn

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        """Applies the hard sigmoid function element-wise."""
        return self.relu(x + 3) / 6

class ChannelSelfAttention(nn.Module):
    def __init__(self, channel, activation=HardSigmoid):
        super(ChannelSelfAttention, self).__init__()
        self.wq = nn.Conv2d(channel, 1, kernel_size=1)
        self.wv = nn.Conv2d(channel, channel//2, kernel_size=1)
        self.wz = nn.Conv2d(channel//2, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Sequential(
            nn.LayerNorm(channel),
            activation()
        )

    def forward(self, x):
        b,c,h,w = x.size()
        
        q = self.softmax(self.wq(x).reshape(b,-1,1)) # b, h*w, 1
        v = self.wv(x).reshape(b,c//2,-1) # b, c//2, h*w
        z = torch.matmul(v,q).unsqueeze(-1) # b, c//2, 1, 1

        channel_weight = self.head(
            self.wz(z).reshape(b,c,1).permute(0,2,1)
        ).reshape(b, c, 1, 1)  # bs, c, 1, 1
        return channel_weight*x

class SpatialSelfAttention(nn.Module):
    def __init__(self, channel, activation=HardSigmoid):
        super(SpatialSelfAttention, self).__init__()
        self.wv=nn.Conv2d(channel, channel//2, kernel_size=1)
        self.wq=nn.Conv2d(channel, channel//2, kernel_size=1)

        self.softmax = nn.Softmax(-1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = activation()

    def forward(self, x):
        b,c,h,w = x.size()
        v = self.wv(x).reshape(b, c//2, -1) # b, c//2, h*w
        q = self.softmax(
            self.pool(
                # avg pool
                self.wq(x)
            ).permute(0,2,3,1).reshape(b,1,c//2)
        ) # b, 1, c//2

        z = torch.matmul(q,v) # (b, 1, c//2) * (b, c//2, h*w) --> (b, 1, h*w)
        spatial_weight = self.sigmoid(z.reshape(b,1,h,w)) # b, 1, h, w
        return spatial_weight*x

#############################################################

class SPSA(nn.Module):
    """Serial PSA"""
    def __init__(self, channel=512):
        super().__init__()
        self.channel_attention = ChannelSelfAttention(channel, activation=nn.Sigmoid)
        self.spatial_attention = SpatialSelfAttention(channel, activation=nn.Sigmoid)
    def forward(self, x):
        return self.channel_attention(self.spatial_attention(x))

class HSPSA(nn.Module):
    """Hard Serial PSA"""
    def __init__(self, channel=512):
        super().__init__()
        self.channel_attention = ChannelSelfAttention(channel, activation=HardSigmoid)
        self.spatial_attention = SpatialSelfAttention(channel, activation=HardSigmoid)
    def forward(self, x):
        return self.channel_attention(self.spatial_attention(x))

#############################################################

class HSPSAFFN(nn.Module):
    """
    Hard Serial Polarized Self Attention module with FFN.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention module, and feed-forward network with channel reduction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = HSPSA(self.c)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Forward pass of the PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
    
class SSPSAFFN(nn.Module):
    """
    Soft Serial Polarized Self Attention module with FFN.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention module, and feed-forward network with channel reduction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = SPSA(self.c)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Forward pass of the PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))