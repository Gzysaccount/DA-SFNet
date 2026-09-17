import pywt  # 小波变换核心库（提供小波基、分解/重构滤波器）
import pywt.data
import torch
from torch import nn
from functools import partial # 偏函数：固定函数部分参数
import torch.nn.functional as F  # 函数式接口（卷积、转置卷积等）

from .conv import Conv # 自定义基础卷积（YOLO系列）
from .block import C2f, C3, Bottleneck # YOLO核心特征模块（CSP瓶颈结构）


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave) # 初始化小波基（如db1、db2)
     # 小波分解滤波器：高低通滤波器（反转，适配卷积的互相关运算）
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
     # 组合二维小波分解滤波器（4个子带：LL/LH/HL/HH）
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1), # LL（低频+低频）
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1), # LH（低频+高频）
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1), # HL（高频+低频）
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0) # HH（高频+高频）

    # 扩展通道维度：适配输入通道数（每个通道用相同小波滤波器）
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 小波重构滤波器：高低通滤波器（反转+翻转，适配转置卷积）
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    # 组合二维小波重构滤波器（与分解对应）
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters
  # 滤波器反转：卷积本质是 “互相关”，需反转滤波器才能等价于小波变换的卷积运算；
  # 4 个子带：二维小波分解将图像分为低频（LL）和三个高频子带（LH/HL/HH）；
  # 通道扩展：repeat(in_size, 1,1,1) 让每个输入通道共享同一小波滤波器（分组卷积用）。

def wavelet_transform(x, filters):
    b, c, h, w = x.shape # b=批次, c=通道, h=高, w=宽
    # 计算填充：保证卷积后尺寸匹配（same padding变种）
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 分组卷积实现小波分解：stride=2 → 尺寸减半（小波分解特性）
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    # 重塑形状：分离4个子带 → (b, c, 4, h//2, w//2)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
    # 核心逻辑：用stride=2的分组卷积替代传统小波分解，输出 4 个子带的张量（便于后续卷积处理）。

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 重塑形状：合并4个子带到通道维度 → (b, c*4, h_half, w_half)
    x = x.reshape(b, c * 4, h_half, w_half)
    # 分组转置卷积：stride=2 → 尺寸还原（小波重构特性）
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
    # 核心逻辑：将分解的 4 个子带合并，通过转置卷积还原原始尺寸，完成小波重构。


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels  # 小波变换要求输入输出通道数一致

        self.in_channels = in_channels
        self.wt_levels = wt_levels # 小波变换的层数（多尺度）
        self.stride = stride
        self.dilation = 1

        # 1. 初始化小波滤波器（不训练，requires_grad=False）
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 2. 偏函数封装小波/逆小波变换（固定滤波器参数）
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        # 3. 基础卷积分支：深度卷积（分组=通道数，轻量化）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1]) # 缩放基础分支

        # 4. 小波分支：每层小波分解对应一个分组卷积+缩放
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        # 5. 空间-频域融合门控（Spatial-Frequency Fusion Gate）
        self.fag_gate = SFFG(in_channels, reduction=8)
        
        # 6. 下采样：如果 stride>1，则在输出端进行下采样
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        # 初始化多尺度小波分解的缓存列表
        x_ll_in_levels = [] # 各层LL子带（低频）
        x_h_in_levels = [] # 各层高频子带（LH/HL/HH）
        shapes_in_levels = [] # 各层输入形状（用于重构时裁剪,处理奇数尺寸）

        curr_x_ll = x # 初始LL子带为输入

        # 步骤1：多尺度小波分解 + 小波分支卷积
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
             # 处理奇数尺寸：填充0，保证stride=2后尺寸为整数
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            # 小波分解
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]  # 提取当前层LL子带（用于下一层分解）
            # 小波分支卷积+缩放
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            # 保存处理后的子带
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

# 步骤2：多尺度小波重构 + 融合
        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            # 融合上一层重构的LL子带
            curr_x_ll = curr_x_ll + next_x_ll
            # 拼接LL和高频子带，重构
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            # 裁剪回原始形状（去掉之前的填充）
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        
        # 步骤3：融合基础卷积分支和小波分支
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        base = self.base_scale(self.base_conv(x)) # 基础卷积分支
        x = self.fag_gate(base, x_tag)  # 自适应门控融合，替代固定相加
        # 步骤4：下采样（如果需要）
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
# 多尺度小波分解：对输入逐层做小波分解，每层 LL 子带继续分解，高频子带缓存；
# 小波分支卷积：对每层分解的 4 个子带做分组卷积 + 缩放；
# 逆序重构融合：从最深层开始重构，逐层融合上一层的 LL 子带；
# 分支融合：小波重构结果与基础深度卷积结果残差融合，增强特征；
# 下采样：支持 stride>1 的下采样（适配网络的尺度压缩）。

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale) # 可学习缩放权重
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x) # 逐元素相乘（广播到整个张量）
 # 作用：自适应调整 “基础卷积分支” 和 “小波分支” 的权重（比如小波分支初始缩放 0.1，避免权重过大）。


class SFFG(nn.Module):
    """Spatial-Frequency Fusion Gate.
    
    Adaptively fuses base convolution features and wavelet-transformed features
    via a lightweight spatial attention mechanism. This allows the network to
    dynamically decide the contribution of frequency-domain information at each
    spatial location.
    """
    
    def __init__(self, in_channels, reduction=8):
        super(SFFG, self).__init__()
        # Ensure reduction does not exceed in_channels
        reduction = min(reduction, in_channels)
        while in_channels // reduction < 1 and reduction > 1:
            reduction -= 1
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 2, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, base, wavelet):
        x = torch.cat([base, wavelet], dim=1)
        attn = self.sigmoid(self.conv(x))  # (B, 2, H, W)
        return attn[:, 0:1, :, :] * base + attn[:, 1:2, :, :] * wavelet

class DSConvWithWT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DSConvWithWT, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
# 深度可分离卷积 = 深度卷积（逐通道） + 逐点卷积（1x1 融合通道），这里用WTConv2d替换普通深度卷积：

class Bottleneck_WT(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1) # 普通卷积降维
        self.cv2 = WTConv2d(c_, c2)   # 小波卷积升维   
        self.add = shortcut and c1 == c2  # 短路连接（残差）

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    # 替换 YOLO 标准 Bottleneck 的第二个卷积为WTConv2d，增强特征提取：

class C3k_WT(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    # C3k_WT：基于C3的小波版本
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # 替换C3内部的Bottleneck为Bottleneck_WT
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WT(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_WT特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_WT(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            C3k_WT(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

if __name__ == '__main__':
    DW = DSConvWithWT(256, 128)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =DW(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
#测试目标：验证DSConvWithWT的输入输出形状是否符合预期（通道从 256→128，尺寸保持 64x64）。