import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSelfAttention(nn.Module):
    """
    Channel-wise self-attention module.

    Args:
        channel (int): Number of input channels.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.Hardsigmoid.

    Attributes:
        query_conv (nn.Conv2d): 1x1 convolution to generate query vectors.
        value_conv (nn.Conv2d): 1x1 convolution to generate value vectors.
        output_conv (nn.Conv2d): 1x1 convolution to generate output vectors.
        softmax (nn.Softmax): Softmax function to normalize attention weights.
        head (nn.Sequential): Sequential module containing LayerNorm and activation function.
    """
    def __init__(self, channel, activation=nn.Hardsigmoid):
        super().__init__()
        self.query_conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.value_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.output_conv = nn.Conv2d(channel // 2, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Sequential(
            nn.LayerNorm(channel),
            activation()
        )

    def forward(self, x):
        """
        Forward pass for channel-wise self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        b, c, h, w = x.size()

        query = self.softmax(self.query_conv(x).view(b, -1, 1))  # b, h*w, 1
        value = self.value_conv(x).view(b, c // 2, -1)  # b, c//2, h*w
        attention_map = torch.matmul(value, query).unsqueeze(-1)  # b, c//2, 1, 1

        channel_weight = self.head(
            self.output_conv(attention_map).view(b, c, 1).permute(0, 2, 1)
        ).view(b, c, 1, 1)  # b, c, 1, 1
        return channel_weight * x

class SpatialSelfAttention(nn.Module):
    """
    Spatial-wise self-attention module.

    Args:
        channel (int): Number of input channels.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.Hardsigmoid.

    Attributes:
        value_conv (nn.Conv2d): 1x1 convolution to generate value vectors.
        query_conv (nn.Conv2d): 1x1 convolution to generate query vectors.
        softmax (nn.Softmax): Softmax function to normalize attention weights.
        pool (nn.AdaptiveAvgPool2d): Adaptive average pooling to reduce spatial dimensions.
        sigmoid (nn.Module): Activation function.
    """
    def __init__(self, channel, activation=nn.Hardsigmoid):
        super().__init__()
        self.value_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.query_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.softmax = nn.Softmax(-1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = activation()

    def forward(self, x):
        """
        Forward pass for spatial-wise self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        b, c, h, w = x.size()
        value = self.value_conv(x).view(b, c // 2, -1)  # b, c//2, h*w
        query = self.softmax(
            self.pool(self.query_conv(x)).permute(0, 2, 3, 1).view(b, 1, c // 2)
        )  # b, 1, c//2

        attention_map = torch.matmul(query, value)  # (b, 1, c//2) * (b, c//2, h*w) --> (b, 1, h*w)
        spatial_weight = self.sigmoid(attention_map.view(b, 1, h, w))  # b, 1, h, w
        return spatial_weight * x

class PSA(nn.Module):
    """
    Parallel Spatial and Channel Self-Attention module.

    Args:
        channel (int, optional): Number of input channels. Defaults to 512.
        psa_type (str, optional): Type of activation function to use. Can be "hard" or "soft". Defaults to "hard".

    Attributes:
        channel_attention (ChannelSelfAttention): Channel-wise self-attention module.
        spatial_attention (SpatialSelfAttention): Spatial-wise self-attention module.
    """
    def __init__(self, channel=512, psa_type="hard"):
        super().__init__()
        activation = nn.Hardsigmoid if psa_type == "hard" else nn.Sigmoid
        self.channel_attention = ChannelSelfAttention(channel, activation)
        self.spatial_attention = SpatialSelfAttention(channel, activation)

    def forward(self, x):
        """
        Forward pass for PSA module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        return self.channel_attention(self.spatial_attention(x))

class ResidualPSA(nn.Module):
    def __init__(self, channel, psa_type="hard"):
        super().__init__()
        self.refinement = PSA(channel, psa_type)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        refined = self.refinement(x)
        return x + torch.tanh(self.gamma) * refined

class PSAFFN(nn.Module):
    """
    Parallel Spatial and Channel Self-Attention Feed-Forward Network.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float, optional): Expansion factor for intermediate channels. Defaults to 0.5.
        psa_type (str, optional): Type of activation function to use. Can be "hard" or "soft". Defaults to "hard".

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (nn.Conv2d): 1x1 convolution to split input tensor.
        cv2 (nn.Conv2d): 1x1 convolution to merge output tensor.
        attn (PSA): Parallel Spatial and Channel Self-Attention module.
        ffn (nn.Sequential): Feed-Forward Network module.
    """
    def __init__(self, c1, c2, e=0.5, psa_type="hard"):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(2 * self.c, c1, 1, bias=False)
        
        self.attn = ResidualPSA(self.c, psa_type) # redusial attn
        
        self.ffn = nn.Sequential(
            nn.Conv2d(self.c, self.c * 2, 1, bias=False),
            nn.BatchNorm2d(self.c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c * 2, self.c, 1, bias=False)
        )

    def forward(self, x):
        """
        Forward pass for PSAFFN module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        a, b = self.cv1(x).chunk(2, dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

# Usage
HSPSAFFN = lambda c1, c2, e=0.5: PSAFFN(c1, c2, e, psa_type="hard")
SSPSAFFN = lambda c1, c2, e=0.5: PSAFFN(c1, c2, e, psa_type="soft")