import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DA_FFT"]


class DA_FFT(nn.Module):
    """Direction-aware local FFT residual refinement for CAD line drawings.

    The module extracts direction-selective, non-DC frequency responses from local
    windows and adds them back to the pretrained feature map through a bounded
    residual gate. The initial perturbation is deliberately small so that a YOLO
    checkpoint remains a valid starting point for an ablation experiment.

    Args:
        channels (int): Number of input/output channels.
        window_size (int): Local FFT window size, a power of two. Default: 16.
        num_groups (int): Groups of the 1x1 post-FFT refinement. Default: 4.
        direction_angles (list[int]): Axis directions in degrees. Default: [0, 90].
        max_residual_scale (float): Maximum magnitude of the residual gate. Default: 0.05.
        initial_residual_scale (float): Initial effective residual gate. Default: 0.005.
        highpass_cutoff (float): Normalized frequency at which the high-pass mask
            reaches roughly 63 percent of its maximum. Default: 0.15.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        num_groups: int = 4,
        direction_angles: Optional[List[int]] = None,
        max_residual_scale: float = 0.05,
        initial_residual_scale: float = 0.005,
        highpass_cutoff: float = 0.15,
    ):
        super().__init__()
        if channels % num_groups:
            raise ValueError(f"channels {channels} must be divisible by num_groups {num_groups}")
        if window_size < 2 or window_size & (window_size - 1):
            raise ValueError(f"window_size must be a power of two and >= 2, got {window_size}")
        if not direction_angles:
            raise ValueError("direction_angles must contain at least one direction")
        if max_residual_scale <= 0:
            raise ValueError(f"max_residual_scale must be > 0, got {max_residual_scale}")
        if not 0 <= initial_residual_scale < max_residual_scale:
            raise ValueError(
                "initial_residual_scale must satisfy "
                f"0 <= initial_residual_scale < max_residual_scale, got "
                f"{initial_residual_scale} and {max_residual_scale}"
            )
        if highpass_cutoff <= 0:
            raise ValueError(f"highpass_cutoff must be > 0, got {highpass_cutoff}")

        self.channels = channels
        self.window_size = window_size
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        self.direction_angles = direction_angles if direction_angles is not None else [0, 90]
        self.num_directions = len(self.direction_angles)
        self.max_residual_scale = float(max_residual_scale)
        self.highpass_cutoff = float(highpass_cutoff)

        # rfft2 has a full vertical spectrum and a non-negative horizontal spectrum.
        # The DC coordinate is exactly zero in ``radius`` and is consequently removed
        # by the high-pass envelope below.
        fy = torch.fft.fftfreq(window_size)
        fx = torch.fft.rfftfreq(window_size)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
        self.register_buffer("phi", torch.atan2(grid_y, grid_x))
        self.register_buffer("radius", torch.sqrt(grid_x.square() + grid_y.square()))

        # Equal softmax logits yield a neutral mixture of the requested CAD axes.
        # The angular bandwidth is parameterized in radians and kept positive with
        # softplus; its initial value is about 35 degrees.
        self.direction_logits = nn.Parameter(torch.zeros(self.num_directions))
        self.angle_offsets = nn.Parameter(torch.zeros(self.num_directions))
        bandwidth_init = 0.61  # approximately 35 degrees
        self.bandwidth_logits = nn.Parameter(
            torch.full((self.num_directions,), math.log(math.expm1(bandwidth_init)))
        )

        # A grouped identity 1x1 convolution retains the frequency residual at
        # initialization, instead of randomly rewriting it. It also lets each channel
        # group learn a light post-FFT projection during training.
        self.refine = nn.Conv2d(channels, channels, 1, groups=num_groups, bias=False)

        # Bounded residual gate: prevents FFT processing from overwriting pretrained
        # YOLO features, while the non-zero initial scale supplies learning signal.
        gamma_init = math.atanh(initial_residual_scale / self.max_residual_scale)
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.refine.weight)
        with torch.no_grad():
            for group in range(self.num_groups):
                offset = group * self.group_channels
                for channel in range(self.group_channels):
                    self.refine.weight[offset + channel, channel, 0, 0] = 1.0

    def _build_directional_masks(self, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Return non-DC directional masks with shape (D, H_fft, W_fft)."""
        phi = self.phi.to(device=device, dtype=torch.float32)
        radius = self.radius.to(device=device, dtype=torch.float32)
        if phi.shape != shape:
            phi = F.interpolate(phi[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]
            radius = F.interpolate(radius[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]

        # This envelope is zero at DC and smoothly attenuates very low frequencies;
        # it keeps DA-FFT from learning a global brightness/feature-scale shortcut.
        highpass = 1.0 - torch.exp(-(radius / self.highpass_cutoff).square())
        masks = []
        for angle, offset, bandwidth_logit in zip(self.direction_angles, self.angle_offsets, self.bandwidth_logits):
            # EMA validation casts module parameters to FP16. Keep every frequency
            # mask calculation in FP32 so it matches ``phi``/``radius`` and remains
            # numerically stable under AMP.
            target = math.radians(angle) + offset.float()
            bandwidth = F.softplus(bandwidth_logit.float()).clamp_min(1e-3)
            # abs(sin(.)) makes the direction axial: theta and theta + pi are equal.
            distance = torch.abs(torch.sin(phi - target))
            masks.append(torch.exp(-0.5 * (distance / bandwidth).square()) * highpass)
        return torch.stack(masks, dim=0)

    def _windowed_fft_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a local direction-selective high-frequency residual in FP32."""
        b, c, h, w = x.shape
        win = min(self.window_size, h, w)
        win = max(2, 2 ** int(math.log2(win)))

        pad_h = (win - h % win) % win
        pad_w = (win - w % win) % win
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_pad = F.pad(x.float(), (0, pad_w, 0, pad_h), mode="reflect")
            _, _, hp, wp = x_pad.shape
            nh, nw = hp // win, wp // win

            x_win = x_pad.reshape(b, c, nh, win, nw, win)
            x_win = x_win.permute(0, 1, 2, 4, 3, 5).reshape(-1, win, win)
            spectrum = torch.fft.rfft2(x_win, norm="ortho")

            masks = self._build_directional_masks(spectrum.shape[-2:], x.device)
            # ``masks`` intentionally stay FP32; explicitly cast the learnable
            # logits because the validation EMA model may store them in FP16.
            direction_weights = F.softmax(self.direction_logits.float(), dim=0)
            mask = torch.einsum("d,dhw->hw", direction_weights, masks)
            residual_win = torch.fft.irfft2(spectrum * mask, s=(win, win), norm="ortho")

            residual = residual_win.reshape(b, c, nh, nw, win, win)
            residual = residual.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hp, wp)
            return residual[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pretrained feature map plus a bounded FFT directional residual."""
        identity, dtype = x, x.dtype
        residual = self._windowed_fft_residual(x)
        # In EMA validation the learned convolution can be FP16 while the FFT path
        # intentionally remains FP32, so match the convolution parameter dtype here.
        residual = self.refine(residual.to(dtype=self.refine.weight.dtype))
        scale = self.max_residual_scale * torch.tanh(self.gamma)
        return (identity + scale * residual).to(dtype)


if __name__ == "__main__":
    model = DA_FFT(128, window_size=16, num_groups=4)
    x = torch.randn(2, 128, 40, 40)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
