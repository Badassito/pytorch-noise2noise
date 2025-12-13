# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.cablock import CABlock, FeedForward
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm


class MDTATransformerBlock(nn.Module):
    """Basic transformer unit combining MDTA and GDFN with skip connections."""

    def __init__(
        self,
        spatial_dims: int,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_use_bias: bool = False,
        flash_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = Norm[Norm.INSTANCE, spatial_dims](dim, affine=layer_norm_use_bias)
        self.attn = CABlock(spatial_dims, dim, num_heads, bias, flash_attention)
        self.norm2 = Norm[Norm.INSTANCE, spatial_dims](dim, affine=layer_norm_use_bias)
        self.ffn = FeedForward(spatial_dims, dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(Convolution):
    """Initial feature extraction using overlapped convolutions."""

    def __init__(self, spatial_dims: int, in_channels: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=3,
            strides=1,
            padding=1,
            bias=bias,
            conv_only=True,
        )


class PixelUnshuffle(nn.Module):
    """Explicit Pixel Unshuffle to avoid MONAI DownSample wrapper ambiguity."""
    def __init__(self, spatial_dims: int, scale: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2:
            return F.pixel_unshuffle(x, self.scale)
        elif self.spatial_dims == 3:
            # (B, C, D, H, W) -> (B, C*s^3, D/s, H/s, W/s)
            b, c, d, h, w = x.shape
            s = self.scale
            x = x.view(b, c, d // s, s, h // s, s, w // s, s)
            x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
            x = x.view(b, c * (s**3), d // s, h // s, w // s)
            return x
        return x


class PixelShuffle(nn.Module):
    """Explicit Pixel Shuffle."""
    def __init__(self, spatial_dims: int, scale: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 2:
            return F.pixel_shuffle(x, self.scale)
        elif self.spatial_dims == 3:
            # (B, C*s^3, D/s, H/s, W/s) -> (B, C, D, H, W)
            b, c_in, d, h, w = x.shape
            s = self.scale
            c_out = c_in // (s**3)
            x = x.view(b, c_out, s, s, s, d, h, w)
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            x = x.view(b, c_out, d * s, h * s, w * s)
            return x
        return x


class Restormer(nn.Module):
    """Restormer: Efficient Transformer for High-Resolution Image Restoration."""

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple[int, ...] = (1, 1, 1, 1),
        heads: tuple[int, ...] = (1, 1, 1, 1),
        num_refinement_blocks: int = 4,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_use_bias: bool = True,
        dual_pixel_task: bool = False,
        flash_attention: bool = False,
    ) -> None:
        super().__init__()

        assert len(num_blocks) > 1, "Number of blocks must be greater than 1"
        assert len(num_blocks) == len(heads), "Number of blocks and heads must be equal"

        self.spatial_dims = spatial_dims
        self.num_steps = len(num_blocks) - 1

        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(spatial_dims, in_channels, dim)

        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        pixel_shuffle_expansion = 2 ** spatial_dims

        # --- Define Encoder Levels ---
        for n in range(self.num_steps):
            current_dim = dim * (2**n)
            next_dim = dim * (2**(n + 1))

            self.encoder_levels.append(
                nn.Sequential(
                    *[
                        MDTATransformerBlock(
                            spatial_dims=spatial_dims,
                            dim=current_dim,
                            num_heads=heads[n],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            layer_norm_use_bias=layer_norm_use_bias,
                            flash_attention=flash_attention,
                        )
                        for _ in range(num_blocks[n])
                    ]
                )
            )

            # Explicit Downsampling: Unshuffle -> Conv
            # 2D Example: 48 ch -> Unshuffle (scale 2) -> 192 ch -> Conv -> 96 ch
            unshuffled_channels = current_dim * pixel_shuffle_expansion

            self.downsamples.append(
                nn.Sequential(
                    PixelUnshuffle(spatial_dims, scale=2),
                    Convolution(
                        spatial_dims=self.spatial_dims,
                        in_channels=unshuffled_channels,
                        out_channels=next_dim,
                        kernel_size=1,
                        bias=bias,
                        conv_only=True,
                    )
                )
            )

        # --- Define Latent Space ---
        latent_dim = dim * (2**self.num_steps)
        self.latent = nn.Sequential(
            *[
                MDTATransformerBlock(
                    spatial_dims=spatial_dims,
                    dim=latent_dim,
                    num_heads=heads[self.num_steps],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_use_bias=layer_norm_use_bias,
                    flash_attention=flash_attention,
                )
                for _ in range(num_blocks[self.num_steps])
            ]
        )

        # --- Define Decoder Levels ---
        for n in reversed(range(self.num_steps)):
            current_dim = dim * (2**n)
            next_dim = dim * (2**(n + 1))

            # Explicit Upsampling: Conv -> Shuffle
            # 2D Example: 96 ch -> Conv -> 192 ch -> Shuffle (scale 2) -> 48 ch
            target_shuffled_channels = current_dim * pixel_shuffle_expansion

            self.upsamples.append(
                nn.Sequential(
                    Convolution(
                        spatial_dims=self.spatial_dims,
                        in_channels=next_dim,
                        out_channels=target_shuffled_channels,
                        kernel_size=1,
                        bias=bias,
                        conv_only=True,
                    ),
                    PixelShuffle(spatial_dims, scale=2)
                )
            )

            # Channel reduction after skip connection concatenation
            # Input: current_dim (upsampled) + current_dim (skip)
            self.reduce_channels.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=current_dim * 2,
                    out_channels=current_dim,
                    kernel_size=1,
                    bias=bias,
                    conv_only=True,
                )
            )

            self.decoder_levels.append(
                nn.Sequential(
                    *[
                        MDTATransformerBlock(
                            spatial_dims=spatial_dims,
                            dim=current_dim,
                            num_heads=heads[n],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            layer_norm_use_bias=layer_norm_use_bias,
                            flash_attention=flash_attention,
                        )
                        for _ in range(num_blocks[n])
                    ]
                )
            )

        # --- Refinement and Output ---
        self.refinement = nn.Sequential(
            *[
                MDTATransformerBlock(
                    spatial_dims=spatial_dims,
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_use_bias=layer_norm_use_bias,
                    flash_attention=flash_attention,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=1,
                bias=bias,
                conv_only=True,
            )

        self.output = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=dim * 2 if self.dual_pixel_task else dim,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            bias=bias,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Restormer."""
        assert all(
            x.shape[-i] > 2**self.num_steps for i in range(1, self.spatial_dims + 1)
        ), "All spatial dimensions should be larger than 2^number_of_step"

        inp_img = x

        # Patch embedding
        x = self.patch_embed(x)
        skip_connections = []

        # Encoding path
        for encoder, downsample in zip(self.encoder_levels, self.downsamples):
            x = encoder(x)
            skip_connections.append(x)
            x = downsample(x)

        # Latent space
        x = self.latent(x)

        # Decoding path
        for idx in range(len(self.decoder_levels)):
            x = self.upsamples[idx](x)
            skip = skip_connections[-(idx + 1)]
            x = torch.concat([x, skip], 1)
            x = self.reduce_channels[idx](x)
            x = self.decoder_levels[idx](x)

        # Final refinement
        x = self.refinement(x)

        if self.dual_pixel_task:
            x = x + self.skip_conv(skip_connections[0])
            x = self.output(x)
        else:
            x = self.output(x)
            # Add global residual connection (Input + Residual)
            if x.shape == inp_img.shape:
                x = x + inp_img

        return x
