import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ============================================
# 1. LGC (Learnable Gabor Convolution) - 基础模块
# ============================================

class LGC(nn.Module):
    """Learnable Gabor Convolution.
    
    基于 GCN 2017 的改进版：将频率f、方向θ、高斯宽度σ 全部参数化，
    通过反向传播优化，而非固定参数。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=7, num_orientations=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        
        # 可学习 Gabor 参数（核心创新：非固定）
        # 每个方向一组参数
        self.theta = nn.Parameter(torch.rand(num_orientations) * math.pi)  # 方向
        self.freq = nn.Parameter(torch.ones(num_orientations) * 0.1)  # 频率
        self.sigma = nn.Parameter(torch.ones(num_orientations) * 2.0)  # 高斯宽度
        self.gamma = nn.Parameter(torch.ones(num_orientations) * 1.0)  # 空间纵横比
        
        # 1x1 卷积用于通道映射
        self.pointwise = nn.Conv2d(in_channels * num_orientations, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def generate_gabor_kernel(self, theta, freq, sigma, gamma, size):
        """生成可微分的 Gabor 核"""
        t = torch.arange(-size//2, size//2 + 1, dtype=torch.float32, device=theta.device)
        x, y = torch.meshgrid(t, t, indexing='ij')
        
        # 旋转坐标
        x_rot = x * torch.cos(theta) + y * torch.sin(theta)
        y_rot = -x * torch.sin(theta) + y * torch.cos(theta)
        
        # Gabor 函数
        gaussian = torch.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * math.pi * freq * x_rot)
        kernel = gaussian * sinusoid
        
        return kernel / (kernel.abs().sum() + 1e-8)  # 归一化
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 为每个方向生成 Gabor 核并卷积
        gabor_responses = []
        for i in range(self.num_orientations):
            kernel = self.generate_gabor_kernel(
                self.theta[i], self.freq[i], self.sigma[i], self.gamma[i], self.kernel_size
            )
            # 扩展为 (out_channels, in_channels, k, k) 格式
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(self.out_channels, c, 1, 1)
            # 分组卷积：每组处理一个方向的响应
            padding = self.kernel_size // 2
            response = F.conv2d(x, kernel, padding=padding, groups=c)
            gabor_responses.append(response)
        
        # 拼接所有方向响应
        gabor_out = torch.cat(gabor_responses, dim=1)  # (B, C*num_orientations, H, W)
        
        # 1x1 卷积融合
        out = self.pointwise(gabor_out)
        out = self.bn(out)
        return out


# ============================================
# 2. CDGF (Cross-Directional Gabor Fusion) - 方向竞争融合
# ============================================

class CDGF(nn.Module):
    """Cross-Directional Gabor Fusion.
    
    核心创新：不是简单相加各方向响应，而是设计一个门控机制，
    让水平/垂直/对角方向的响应根据局部结构自适应竞争融合。
    """
    
    def __init__(self, channels):
        super().__init__()
        # 方向竞争门控
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels // 8, 1),  # 3个方向响应
            nn.BatchNorm2d(channels // 8),
            nn.SiLU(),
            nn.Conv2d(channels // 8, 3, 1),
            nn.Softmax(dim=1)  # 3方向竞争权重
        )
        
        # 方向间交互（水平响应对垂直响应的抑制/增强）
        self.cross_interaction = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        
    def forward(self, h_resp, v_resp, d_resp):
        """
        h_resp: 水平Gabor响应 (B, C, H, W)
        v_resp: 垂直Gabor响应 (B, C, H, W)
        d_resp: 对角Gabor响应 (B, C, H, W)
        """
        # 拼接
        concat = torch.cat([h_resp, v_resp, d_resp], dim=1)  # (B, 3C, H, W)
        
        # 竞争权重
        weights = self.gate(concat)  # (B, 3, H, W)
        
        # 方向间交互（考虑方向间的增强/抑制关系）
        interacted = self.cross_interaction(concat)  # (B, C, H, W)
        
        # 加权融合：竞争 + 交互
        out = (weights[:, 0:1] * h_resp + 
               weights[:, 1:2] * v_resp + 
               weights[:, 2:3] * d_resp) * interacted
        
        return out


# ============================================
# 3. GEGDK (Gabor-Energy-Guided Dynamic Kernel) - 核心创新
# ============================================

class GEGDK(nn.Module):
    """Gabor-Energy-Guided Dynamic Kernel.
    
    核心创新：不是简单拼接 LGC 和 DADK，而是设计一个可学习的
    映射函数，将 Gabor 方向响应能量直接转换为动态核参数。
    
    物理驱动闭环：
    Gabor 提取方向特征 → 能量映射决定感知策略 → 动态核执行空间感知
    """
    
    def __init__(self, channels, kernel_sizes=[3, 5, 7, 9], num_orientations=4):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.num_ks = len(kernel_sizes)
        self.num_orientations = num_orientations
        
        # 1. LGC：提取方向特征
        self.lgc = LGC(channels, channels // 2, kernel_size=7, num_orientations=num_orientations)
        
        # 2. Gabor 能量提取：将方向响应压缩为能量图
        self.energy_extract = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.SiLU(),
        )
        
        # ★ 核心创新：可学习映射 M(E) → 核大小选择权重
        # 输入：3方向能量 (E_h, E_v, E_d) + 压缩特征
        self.kernel_selector = nn.Sequential(
            nn.Conv2d(channels // 8 + 3, channels // 16, 1),
            nn.BatchNorm2d(channels // 16),
            nn.SiLU(),
            nn.Conv2d(channels // 16, self.num_ks * 2, 1),  # *2 for k_h, k_w
        )
        
        # 3. 多尺度方向核库（深度可分离）
        # 水平核（宽扁）
        self.conv_h = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, k), padding=(0, k//2), groups=channels)
            for k in kernel_sizes
        ])
        # 垂直核（高瘦）
        self.conv_v = nn.ModuleList([
            nn.Conv2d(channels, channels, (k, 1), padding=(k//2, 0), groups=channels)
            for k in kernel_sizes
        ])
        
        # 4. 输出融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        
        # 5. 残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        # 步骤1：LGC 提取方向特征
        gabor_feat = self.lgc(x)  # (B, C/2, H, W)
        
        # 步骤2：提取 Gabor 能量
        energy_feat = self.energy_extract(gabor_feat)  # (B, C/8, H, W)
        
        # 计算三方向能量（从 num_orientations 个方向聚合为 3 个主方向）
        # 假设前 num_orientations//2 为水平/对角，后 num_orientations//2 为垂直/对角
        c_per_dir = gabor_feat.size(1) // self.num_orientations
        E_h = gabor_feat[:, :c_per_dir * 2].abs().mean(dim=1, keepdim=True)  # 水平+对角
        E_v = gabor_feat[:, c_per_dir * 2:].abs().mean(dim=1, keepdim=True)  # 垂直+对角
        E_d = gabor_feat[:, c_per_dir:c_per_dir * 3].abs().mean(dim=1, keepdim=True)  # 纯对角
        E = torch.cat([E_h, E_v, E_d], dim=1)  # (B, 3, H, W)
        
        # 步骤3：★ 可学习映射 M(E) → 核选择权重
        selector_input = torch.cat([energy_feat, E], dim=1)  # (B, C/8 + 3, H, W)
        weights = self.kernel_selector(selector_input)  # (B, num_ks*2, H, W)
        w_h, w_v = weights[:, :self.num_ks], weights[:, self.num_ks:]  # 各 (B, num_ks, H, W)
        w_h = F.softmax(w_h, dim=1)  # 水平核选择权重（像素级）
        w_v = F.softmax(w_v, dim=1)  # 垂直核选择权重（像素级）
        
        # 步骤4：用权重对多尺度核进行加权融合（像素级动态核）
        h_out = sum(w_h[:, i:i+1] * conv(x) for i, conv in enumerate(self.conv_h))
        v_out = sum(w_v[:, i:i+1] * conv(x) for i, conv in enumerate(self.conv_v))
        
        # 步骤5：融合输出
        out = self.fusion(torch.cat([h_out, v_out], dim=1))
        
        return x + self.residual_scale * out  # 残差


# ============================================
# 4. DS-C3k2_GEGDK - 将 GEGDK 集成到 YOLOv13 的 DS-C3k2 结构
# ============================================

class DSConv(nn.Module):
    """深度可分离卷积 (YOLOv13 标准模块)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class DS_Bottleneck_GEGDK(nn.Module):
    """深度可分离 Bottleneck + GEGDK.
    
    YOLOv13 的 DS-C3k2 中的 Bottleneck，将标准卷积替换为 GEGDK。
    """
    
    def __init__(self, channels, shortcut=True, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.shortcut = shortcut
        
        # 第一个 DSConv
        self.conv1 = DSConv(channels, channels, 3, 1)
        
        # ★ 核心：将第二个 DSConv 替换为 GEGDK
        self.gegdk = GEGDK(channels, kernel_sizes=kernel_sizes)
        
        # 残差缩放（用于训练稳定性）
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1) if not shortcut else None
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.gegdk(out)
        
        if self.shortcut:
            return x + out  # 标准残差
        else:
            return x + self.residual_scale * out  # 缩放残差


class DS_C3k2_GEGDK(nn.Module):
    """YOLOv13 DS-C3k2 模块的 GEGDK 版本.
    
    将 DS-C3k2 中的 Bottleneck 替换为 DS_Bottleneck_GEGDK。
    
    结构：
    Input → Split → DSConv(cv1) → n×DS_Bottleneck_GEGDK → DSConv(cv2) → Concat → DSConv(cv3) → Output
    """
    
    def __init__(self, in_channels, out_channels, n=2, shortcut=True, e=0.5, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.out_channels = out_channels
        self.n = n  # Bottleneck 数量
        
        # 通道分割参数
        c_ = int(out_channels * e)  # 隐藏通道数
        
        # 输入分割卷积：将输入分为两路
        self.cv1 = DSConv(in_channels, 2 * c_, 1, 1)  # 1x1 分割
        
        # Bottleneck 序列
        self.m = nn.ModuleList([
            DS_Bottleneck_GEGDK(c_, shortcut=shortcut, kernel_sizes=kernel_sizes)
            for _ in range(n)
        ])
        
        # 输出融合
        self.cv2 = DSConv((2 + n) * c_, out_channels, 1, 1)
        
    def forward(self, x):
        # 1. 分割输入
        y = list(self.cv1(x).chunk(2, dim=1))  # 分为 y[0], y[1]
        
        # 2. 通过 GEGDK Bottleneck 序列
        for bottleneck in self.m:
            y.append(bottleneck(y[-1]))
        
        # 3. 拼接所有分支
        y = torch.cat(y, dim=1)
        
        # 4. 输出融合
        return self.cv2(y)


# ============================================
# 5. HGA (Hypergraph-Gabor Association) - 方向驱动的超图关联
# ============================================

class HGA(nn.Module):
    """Hypergraph-Gabor Association.
    
    核心创新：在 Neck 处用超图建模图元关联（墙-门-窗），
    超边权重由 Gabor 方向响应的相似性计算，而非随机学习。
    
    物理意义：方向相似的区域（如都是水平响应）被关联到同一超边，
    实现"局部方向感知 → 全局关系建模"。
    """
    
    def __init__(self, channels, num_hyperedges=16, num_orientations=4):
        super().__init__()
        self.channels = channels
        self.num_hyperedges = num_hyperedges
        
        # 超边生成：根据特征自适应生成超边隶属度
        self.hyperedge_gen = nn.Sequential(
            nn.Conv2d(channels, num_hyperedges, 1),  # 每个像素属于 num_he 个超边的权重
            nn.Sigmoid()
        )
        
        # 方向特征投影（用于关联计算）
        self.direction_proj = nn.Conv2d(channels, channels // 4, 1)
        
        # 超边特征变换
        self.hyperedge_transform = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(),
        )
        
        # 顶点更新门控
        self.update_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
        
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, gabor_direction_map=None):
        """
        x: 特征图 (B, C, H, W)
        gabor_direction_map: Gabor方向响应图 (B, 3, H, W)，可选
        
        如果提供 gabor_direction_map，超边权重由方向相似性调制；
        否则仅使用特征自适应生成。
        """
        b, c, h, w = x.shape
        n_vertices = h * w
        
        # 1. 超边隶属度：每个像素属于每个超边的权重
        hyperedge_weights = self.hyperedge_gen(x)  # (B, num_he, H, W)
        
        # 如果提供了方向图，用方向相似性调制超边权重
        if gabor_direction_map is not None:
            # 方向相似性：同方向的像素更应该在同一超边
            dir_sim = torch.exp(-((gabor_direction_map.unsqueeze(2) - gabor_direction_map.unsqueeze(1)) ** 2).mean(dim=1, keepdim=True))
            # 调制超边权重
            hyperedge_weights = hyperedge_weights * dir_sim  # 广播: (B, num_he, H, W)
            hyperedge_weights = hyperedge_weights / (hyperedge_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 2. 方向投影特征
        dir_feat = self.direction_proj(x)  # (B, C/4, H, W)
        
        # 3. 展平为顶点
        x_flat = x.reshape(b, c, n_vertices)  # (B, C, N)
        dir_flat = dir_feat.reshape(b, c//4, n_vertices)  # (B, C/4, N)
        he_flat = hyperedge_weights.reshape(b, self.num_hyperedges, n_vertices)  # (B, num_he, N)
        
        # 4. 超图关联矩阵 H (N × num_he)：顶点-超边关联
        H = he_flat.permute(0, 2, 1)  # (B, N, num_he)
        
        # 5. 超图卷积：消息传递
        # 5.1 顶点 → 超边（聚合）
        hyperedge_features = torch.bmm(x_flat, H)  # (B, C, num_he)
        hyperedge_features = hyperedge_features / (H.sum(dim=1, keepdim=True).permute(0, 2, 1) + 1e-8)  # 平均池化
        
        # 5.2 超边变换
        hyperedge_features = self.hyperedge_transform(hyperedge_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # 5.3 超边 → 顶点（分发）
        vertex_update = torch.bmm(hyperedge_features, H.permute(0, 2, 1))  # (B, C, N)
        vertex_update = vertex_update.reshape(b, c, h, w)
        
        # 6. 门控更新
        gate = self.update_gate(torch.cat([x, vertex_update], dim=1))
        out = gate * vertex_update + (1 - gate) * x
        
        return x + self.residual_scale * out  # 残差


# ============================================
# 6. DCR Loss (Directional Consistency Regularization) - 方向一致性正则化
# ============================================

class DCRLoss(nn.Module):
    """Directional Consistency Regularization.
    
    核心创新：利用 CAD 图纸的轴对齐先验，强制 LGC 的 θ 参数收敛到 0° 或 90°。
    
    物理意义：CAD 图纸的墙、门、窗都是水平或垂直的，
    Gabor 的方向参数应该反映这个物理事实。
    """
    
    def __init__(self, lambda_dcr=0.01, target_angles=[0, math.pi/2]):
        super().__init__()
        self.lambda_dcr = lambda_dcr
        self.target_angles = target_angles  # 目标角度：0° 和 90°
        
    def forward(self, theta_params):
        """
        theta_params: LGC 中所有 θ 参数，shape (num_kernels,) 或 (num_groups, num_orientations)
        
        返回：方向一致性正则化损失
        """
        # 将 θ 映射到 [0, π/2]（因为 180° 和 0° 是同一个方向，90° 和 -90° 也是）
        theta_mod = theta_params % math.pi
        theta_mod = torch.min(theta_mod, math.pi - theta_mod)  # 对称到 [0, π/2]
        
        # 计算到 0° 和 90° 的距离
        dist_to_0 = theta_mod  # 距离 0°
        dist_to_90 = torch.abs(theta_mod - math.pi / 2)  # 距离 90°
        
        # 取最小距离（鼓励靠近 0° 或 90° 中的任意一个）
        min_dist = torch.min(dist_to_0, dist_to_90)
        
        # 正则化损失：L2 形式
        loss = (min_dist ** 2).mean()
        
        return loss * self.lambda_dcr
    
    def get_direction_distribution(self, theta_params):
        """可视化辅助：返回当前 θ 参数的分布统计"""
        theta_mod = theta_params % math.pi
        theta_mod = torch.min(theta_mod, math.pi - theta_mod)
        
        # 统计靠近 0° 和 90° 的比例
        near_0 = (theta_mod < math.pi / 4).float().mean().item()
        near_90 = (theta_mod >= math.pi / 4).float().mean().item()
        
        return {
            'near_0_deg': near_0,
            'near_90_deg': near_90,
            'mean_theta_deg': theta_mod.mean().item() * 180 / math.pi,
        }


# ============================================
# 7. 完整损失函数（集成 DCR）
# ============================================

class CADDetectionLoss(nn.Module):
    """CAD 图纸检测损失 = 标准检测损失 + DCR 方向一致性正则化"""
    
    def __init__(self, detection_loss_fn, lambda_dcr=0.01):
        super().__init__()
        self.detection_loss = detection_loss_fn  # 例如 YOLO 的 CIoU + BCE
        self.dcr_loss = DCRLoss(lambda_dcr=lambda_dcr)
        self.lambda_dcr = lambda_dcr
        
    def forward(self, predictions, targets, lgc_theta_params=None):
        """
        predictions: 检测器输出
        targets: 真实标注
        lgc_theta_params: 所有 LGC 模块的 θ 参数列表（用于 DCR）
        """
        # 1. 标准检测损失
        det_loss = self.detection_loss(predictions, targets)
        
        # 2. DCR 损失（如果提供了 θ 参数）
        dcr_loss = torch.tensor(0.0, device=det_loss.device)
        if lgc_theta_params is not None and self.lambda_dcr > 0:
            all_theta = torch.cat([t.view(-1) for t in lgc_theta_params])
            dcr_loss = self.dcr_loss(all_theta)
        
        # 3. 总损失
        total_loss = det_loss + dcr_loss
        
        return {
            'total_loss': total_loss,
            'detection_loss': det_loss,
            'dcr_loss': dcr_loss,
        }


# ============================================
# 8. YOLOv13 配置注册辅助函数
# ============================================

def register_dadk_lgc_modules():
    """注册所有 DADK+LGC 模块到 Ultralytics 解析器"""
    from ultralytics.nn.tasks import parse_model
    
    # 模块映射表
    module_map = {
        'DS_C3k2_GEGDK': DS_C3k2_GEGDK,
        'GEGDK': GEGDK,
        'HGA': HGA,
        'LGC': LGC,
        'CDGF': CDGF,
    }
    
    return module_map


# ============================================
# 9. 测试
# ============================================

if __name__ == '__main__':
    # 测试 GEGDK
    print("=" * 60)
    print("测试 GEGDK")
    print("=" * 60)
    x = torch.randn(2, 256, 64, 64)
    gegdk = GEGDK(256)
    out = gegdk(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in gegdk.parameters()):,}")
    
    # 测试 DS_C3k2_GEGDK
    print("\n" + "=" * 60)
    print("测试 DS_C3k2_GEGDK")
    print("=" * 60)
    x = torch.randn(2, 256, 64, 64)
    ds_c3k2 = DS_C3k2_GEGDK(256, 256, n=2)
    out = ds_c3k2(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in ds_c3k2.parameters()):,}")
    
    # 测试 HGA
    print("\n" + "=" * 60)
    print("测试 HGA")
    print("=" * 60)
    x = torch.randn(2, 256, 32, 32)
    gabor_map = torch.randn(2, 3, 32, 32)
    hga = HGA(256)
    out = hga(x, gabor_map)
    print(f"输入: {x.shape} + 方向图 {gabor_map.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in hga.parameters()):,}")
    
    # 测试 DCR Loss
    print("\n" + "=" * 60)
    print("测试 DCR Loss")
    print("=" * 60)
    theta = torch.tensor([0.1, 1.5, 0.05, 1.4, 0.8])  # 混合角度
    dcr = DCRLoss(lambda_dcr=1.0)
    loss = dcr(theta)
    print(f"θ 参数: {theta * 180 / math.pi} 度")
    print(f"DCR Loss: {loss.item():.6f}")
    print(f"分布统计: {dcr.get_direction_distribution(theta)}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
