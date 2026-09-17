# import math
#
# import numpy as np
# import torch
# import torch.nn as nn
#
# class TeLU(nn.Module):
#     def __init__(self):
#         """
#         Init method.
#         """
#         super().__init__()
#
#     def forward(self, input):
#         """
#         Forward pass of the function.
#         """
#         return input * torch.tanh( torch.exp(input) )
#
#
# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
#
#
# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#         self.act = TeLU()
#
#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         """Apply convolution and activation without batch normalization."""
#         return self.act(self.conv(x))


import math
import numpy as np
import torch
import torch.nn as nn

# https://arxiv.org/pdf/2412.20269

class TeLU(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh(torch.exp(input))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv_TeLU(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.act = TeLU()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


def main():
    """
    测试TeLU激活函数和Conv层的功能
    """
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 测试TeLU激活函数
    print("=== 测试TeLU激活函数 ===")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"输入张量: {x}")
    telu = TeLU()
    y = telu(x)
    print(f"TeLU输出: {y}")

    # 测试Conv层
    print("\n=== 测试Conv层 ===")
    # 创建一个批次大小为1，通道数为3，高度和宽度为32的随机输入张量
    input_tensor = torch.randn(1, 3, 32, 32)
    print(f"输入张量形状: {input_tensor.shape}")

    # 创建一个卷积层，输入通道3，输出通道16，卷积核大小3
    conv_layer = Conv_TeLU(c1=3, c2=16, k=3, s=1)

    # 前向传播
    output_tensor = conv_layer(input_tensor)
    print(f"输出张量形状: {output_tensor.shape}")



if __name__ == "__main__":
    main()
