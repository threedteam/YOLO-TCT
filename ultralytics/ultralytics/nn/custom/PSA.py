import torch
import torch.nn as nn

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        """Applies the hard sigmoid function element-wise."""
        return self.relu(x + 3) / 6

class DynamicHardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=False)
        self.slope = nn.Parameter(torch.ones(1))  # Learnable slope parameter

    def forward(self, x):
        return self.relu(self.slope * x + 3) / 6

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

class CPSA(nn.Module):
    """串行PSA"""
    def __init__(self, channel=512):
        super().__init__()
        self.channel_attention = ChannelSelfAttention(channel, activation=nn.Sigmoid)
        self.spatial_attention = SpatialSelfAttention(channel, activation=nn.Sigmoid)
    def forward(self, x):
        return self.channel_attention(self.spatial_attention(x))

class HCPSA(nn.Module):
    """串行PSA+HardSigmoid"""
    def __init__(self, channel=512):
        super().__init__()
        self.channel_attention = ChannelSelfAttention(channel, activation=HardSigmoid)
        self.spatial_attention = SpatialSelfAttention(channel, activation=HardSigmoid)
    def forward(self, x):
        return self.channel_attention(self.spatial_attention(x))

#############################################################


class PSA(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.channel_attention = ChannelSelfAttention(channel, activation=nn.Sigmoid)
        self.spatial_attention = SpatialSelfAttention(channel, activation=nn.Sigmoid)

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        return channel_out + spatial_out
    
class HPSA(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.channel_attention = ChannelSelfAttention(channel, activation=HardSigmoid)
        self.spatial_attention = SpatialSelfAttention(channel, activation=HardSigmoid)

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        return channel_out + spatial_out
    
#############################################################