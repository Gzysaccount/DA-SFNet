import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

__all__ = ["MST_LADA"]


class MST_LADA(nn.Module):
    """Multi-Scale Structure Tensor Local Attention with Directional Awareness.

    This module uses structure tensor to compute local direction fields, then generates
directional attention weights that enhance features along dominant line directions
(horizontal/vertical for CAD drawings). It operates at multiple scales for robust
    direction estimation across different object sizes.

    Args:
        channels (int): Number of input/output channels (must be same).
        window_size (int): Base window size for structure tensor computation. Default: 3.
        reduction (int): Channel reduction ratio for the lightweight direction branch. Default: 8.
        num_scales (int): Number of scales for multi-scale structure tensor pooling. Default: 3.
        max_residual_scale (float): Maximum absolute residual gain. Default: 0.05.
        initial_residual_scale (float): Initial effective residual gain. Default: 0.01.
        edge_init_scale (float): Initial strength of the zero-mean directional high-pass kernels. Default: 0.1.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 3,
        reduction: int = 8,
        num_scales: int = 3,
        max_residual_scale: float = 0.05,
        initial_residual_scale: float = 0.01,
        edge_init_scale: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        if max_residual_scale <= 0:
            raise ValueError(f"max_residual_scale must be > 0, got {max_residual_scale}")
        if edge_init_scale < 0:
            raise ValueError(f"edge_init_scale must be >= 0, got {edge_init_scale}")
        if not 0 <= initial_residual_scale < max_residual_scale:
            raise ValueError(
                "initial_residual_scale must satisfy "
                f"0 <= initial_residual_scale < max_residual_scale, got "
                f"{initial_residual_scale} and {max_residual_scale}"
            )

        self.window_size = window_size
        self.num_scales = num_scales
        self.edge_init_scale = float(edge_init_scale)

        # --- Sobel filters for gradient computation (fixed, non-learnable) ---
        # Register as buffers so they move to correct device with .to(device)
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # --- Direction branch: channel reduction for lightweight computation ---
        reduced_channels = max(channels // reduction, 8)
        self.dir_proj = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.dir_norm = nn.GroupNorm(min(8, reduced_channels), reduced_channels)
        self.dir_act = nn.SiLU(inplace=True)

        # --- Multi-scale pooling for structure tensor ---
        # Use genuinely different receptive fields, e.g. [3, 5, 7] for the defaults.
        self.scale_kernel_sizes = [window_size + 2 * i for i in range(num_scales)]
        self.pools = nn.ModuleList(
            [nn.AvgPool2d(k, stride=1, padding=k // 2) for k in self.scale_kernel_sizes]
        )

        # --- Direction encoding: encode [cos(2θ), sin(2θ), A] into attention weights ---
        # Output: 2 channels for horizontal and vertical attention weights
        self.dir_encode = nn.Sequential(
            nn.Conv2d(reduced_channels * num_scales * 3, reduced_channels, 1, bias=False),
            nn.GroupNorm(min(8, reduced_channels), reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, 2, 1, bias=False),  # [H_weight, V_weight]
        )

        # --- Channel-wise attention (SE-like) ---
        self.se_fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.se_fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

        # --- Directional spatial attention ---
        # Compute H and V attention maps separately based on direction weights
        self.spatial_h = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False)
        self.spatial_v = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False)

        # --- Residual scaling parameter ---
        # Keep the residual bounded while giving the refinement path a small,
        # non-zero learning signal from the first optimizer step.
        self.max_residual_scale = float(max_residual_scale)
        gamma_init = math.atanh(initial_residual_scale / self.max_residual_scale)
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32), requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Start the learned gates from neutral logits. Initialize the directional
        # filters as weak zero-sum second derivatives: they respond to CAD edges
        # but have no DC response, so they cannot take the shortcut of globally
        # shrinking or amplifying the pretrained feature map.
        nn.init.zeros_(self.dir_encode[-1].weight)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.spatial_h.weight)
        nn.init.zeros_(self.spatial_v.weight)
        with torch.no_grad():
            high_pass = torch.tensor(
                [0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0],
                dtype=self.spatial_h.weight.dtype,
                device=self.spatial_h.weight.device,
            ) * self.edge_init_scale
            self.spatial_h.weight[:, 0, 0, :] = high_pass
            self.spatial_v.weight[:, 0, :, 0] = high_pass

    @staticmethod
    def _zero_mean_depthwise_conv(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        """Apply a depthwise directional filter after removing its DC component."""
        weight = conv.weight - conv.weight.mean(dim=(-2, -1), keepdim=True)
        return F.conv2d(
            x,
            weight,
            bias=None,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    def _compute_structure_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute structure tensor components Ixx, Ixy, Iyy using Sobel filters.

        Args:
            x: (B, C, H, W) reduced feature map.

        Returns:
            Ixx, Ixy, Iyy: each (B, C, H, W)
        """
        _, c, _, _ = x.shape
        # Merely calling x.float() is not enough under an outer CUDA autocast
        # context: eligible convolutions may still be cast back to FP16. Disable
        # autocast explicitly for Sobel filtering and all squared products.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            x_pad = F.pad(x_f, (1, 1, 1, 1), mode="reflect")
            sobel_x = self.sobel_x.float().repeat(c, 1, 1, 1)
            sobel_y = self.sobel_y.float().repeat(c, 1, 1, 1)
            Ix = F.conv2d(x_pad, sobel_x, groups=c, padding=0)
            Iy = F.conv2d(x_pad, sobel_y, groups=c, padding=0)

            Ixx = Ix.square()
            Ixy = Ix * Iy
            Iyy = Iy.square()
        return Ixx, Ixy, Iyy

    def _eigen_decomposition_2x2(self, Ixx: torch.Tensor, Ixy: torch.Tensor, Iyy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eigen-decomposition of 2x2 symmetric structure tensor.

        For each pixel: S = [[Ixx, Ixy], [Ixy, Iyy]]
        trace = Ixx + Iyy
        det = Ixx*Iyy - Ixy^2
        λ_max = (trace + sqrt(trace^2 - 4*det)) / 2
        λ_min = (trace - sqrt(trace^2 - 4*det)) / 2
        θ = 0.5 * arctan2(2*Ixy, Ixx - Iyy)

        Returns:
            θ: orientation in radians, (B, C, H, W)
            λ_max: max eigenvalue, (B, C, H, W)
            λ_min: min eigenvalue, (B, C, H, W)
        """
        # Keep the complete eigensystem calculation in FP32. Subtracting
        # sqrt(eps) is important: a flat region must have zero eigenvalue gap,
        # rather than the artificial 0.001 gap produced by sqrt(0 + eps).
        eps = 1e-6
        with torch.autocast(device_type=Ixx.device.type, enabled=False):
            Ixx = Ixx.float()
            Ixy = Ixy.float()
            Iyy = Iyy.float()
            trace = Ixx + Iyy
            discriminant = (Ixx - Iyy).square() + 4.0 * Ixy.square()
            sqrt_disc = (torch.sqrt(discriminant.clamp_min(0.0) + eps) - math.sqrt(eps)).clamp_min(0.0)
            λ_max = (trace + sqrt_disc) / 2.0
            λ_min = (trace - sqrt_disc) / 2.0
            θ = 0.5 * torch.atan2(2.0 * Ixy, Ixx - Iyy + eps)
        return θ, λ_max, λ_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MST-LADA.

        Args:
            x: (B, C, H, W) input feature map.

        Returns:
            (B, C, H, W) directionally enhanced feature map.
        """
        # The structure-tensor/eigendecomposition path explicitly runs in FP32;
        # ordinary learned convolutions may still benefit from AMP.
        dtype = x.dtype
        identity = x
        _, _, h, w = x.shape

        # --- Step 1: Channel reduction for lightweight direction computation ---
        x_dir = self.dir_act(self.dir_norm(self.dir_proj(x)))

        # --- Step 2: Multi-scale structure tensor computation ---
        scale_features = []
        for pool in self.pools:
            # Pool the reduced features for multi-scale estimation
            x_pooled = pool(x_dir)
            Ixx, Ixy, Iyy = self._compute_structure_tensor(x_pooled)
            # Upsample back to original resolution
            Ixx = F.interpolate(Ixx, size=(h, w), mode="bilinear", align_corners=False)
            Ixy = F.interpolate(Ixy, size=(h, w), mode="bilinear", align_corners=False)
            Iyy = F.interpolate(Iyy, size=(h, w), mode="bilinear", align_corners=False)

            θ, λ_max, λ_min = self._eigen_decomposition_2x2(Ixx, Ixy, Iyy)
            # Coherence must be 0 on flat regions and remain in [0, 1].
            # Keep this direction math in FP32 as well.
            with torch.autocast(device_type=x.device.type, enabled=False):
                anisotropy = ((λ_max - λ_min) / (λ_max + λ_min + 1e-6)).clamp(0.0, 1.0)
                # Encode direction as [cos(2θ), sin(2θ), A]
                cos2θ = torch.cos(2.0 * θ)
                sin2θ = torch.sin(2.0 * θ)
            scale_features.append(torch.cat([cos2θ, sin2θ, anisotropy], dim=1))

        # Concatenate all scale features: (B, C_reduced * num_scales * 3, H, W)
        dir_input = torch.cat(scale_features, dim=1)

        # --- Step 3: Direction encoding to attention weights ---
        # Validation uses a FP16 EMA model without the training autocast context.
        # The structure-tensor path above intentionally returns FP32, so cast only
        # its completed features to the learned encoder's dtype before Conv2d.
        dir_input = dir_input.to(dtype=self.dir_encode[0].weight.dtype)
        dir_weights = self.dir_encode(dir_input)  # (B, 2, H, W)
        # Zero logits are neutral (gain=1), while the learned range [0, 2]
        # supports both suppression and enhancement of each direction.
        h_weight = 2.0 * torch.sigmoid(dir_weights[:, 0:1, :, :])  # (B, 1, H, W)
        v_weight = 2.0 * torch.sigmoid(dir_weights[:, 1:2, :, :])  # (B, 1, H, W)

        # --- Step 4: Channel-wise attention (SE-like) ---
        se = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        se = self.se_fc2(F.silu(self.se_fc1(se)))  # (B, C, 1, 1)
        # Center the channel gate at 1.0 so its zero-logit initialization is
        # neutral rather than suppressing every channel by 50%.
        channel_attn = 2.0 * torch.sigmoid(se)

        # --- Step 5: Directional spatial attention ---
        # Apply horizontal and vertical convolutions, weighted by direction
        attn_h = self._zero_mean_depthwise_conv(x, self.spatial_h) * h_weight  # (B, C, H, W)
        attn_v = self._zero_mean_depthwise_conv(x, self.spatial_v) * v_weight  # (B, C, H, W)
        # Average the two directional branches to keep their combined magnitude
        # independent of whether both gates are active.
        spatial_attn = 0.5 * (attn_h + attn_v)

        # --- Step 6: Combine and residual ---
        # The zero-mean spatial kernels make this a true directional refinement,
        # not a learnable global rescaling of the pretrained feature map.
        out = spatial_attn * channel_attn
        residual_scale = self.max_residual_scale * torch.tanh(self.gamma)
        return (identity + residual_scale * out).to(dtype)


if __name__ == "__main__":
    # Quick test
    model = MST_LADA(256, window_size=3, reduction=8, num_scales=3)
    x = torch.randn(2, 256, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
