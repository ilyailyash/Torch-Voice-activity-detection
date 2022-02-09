import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 2),
            stride=(2, 1),
            padding=(2, 1),
            **kwargs
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    a = torch.rand(2, 1, 19, 200)
    l1 = CausalConvBlock(1, 20, kernel_size=(3, 2), stride=(2, 1), padding=(0, 1),)
    l2 = CausalConvBlock(20, 40, kernel_size=(3, 2), stride=(1, 1), padding=1, )
    l3 = CausalConvBlock(40, 40, kernel_size=(3, 2), stride=(2, 1), padding=(0, 1), )
    l4 = CausalConvBlock(40, 40, kernel_size=(3, 2), stride=(1, 1), padding=1, )
    print(l1(a).shape)
    print(l4(l3(l2(l1(a)))).shape)
