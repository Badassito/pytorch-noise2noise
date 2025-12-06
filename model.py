from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from torch.optim import lr_scheduler
import numpy as np
import math

from monai.networks.nets import SwinUNETR, BasicUNetPlusPlus
from monai.networks.layers.factories import Conv
from monai.networks.layers import DropPath, trunc_normal_, Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, UpSample
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import, is_sqrt


# Removed UNet, ResNet, Discriminator

def init_weights(net, init_type='normal', init_gain=0.02):
	"""Initialize network weights.

	Parameters:
		net (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		init_gain (float)	-- scaling factor for normal, xavier and orthogonal.

	We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
	work better for some applications. Feel free to try yourself.
	"""
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
	Parameters:
		net (network)	  -- the network to be initialized
		init_type (str)	-- the name of an initialization method: normal | xavier | kaiming | orthogonal
		gain (float)	   -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

	Return an initialized network.
	"""
	if gpu_ids:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	init_weights(net, init_type, init_gain=init_gain)
	if gpu_ids:
	    assert(torch.cuda.is_available())
	    if len(gpu_ids) > 1:
	        net.to(gpu_ids[0])
	        net = torch.nn.DataParallel(net, gpu_ids)
	    else:
	        net.to(gpu_ids[0])
	return net

class SwinUNETRPlusPlus(nn.Module):
	
	patch_size: Final[int] = 2

	def __init__(
		self,
		img_size: Sequence[int] | int,
		in_channels: int,
		out_channels: int,
		depths: Sequence[int] = (2, 2, 2, 2),
		num_heads: Sequence[int] = (3, 6, 12, 24),
		feature_size: int = 24,
		norm_name: tuple | str = "instance",
		drop_rate: float = 0.0,
		attn_drop_rate: float = 0.0,
		dropout_path_rate: float = 0.0,
		normalize: bool = True,
		use_checkpoint: bool = False,
		spatial_dims: int = 3,
		downsample="merging",
		use_v2=False,
		deep_supervision: bool = False,
		act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
		norm: str | tuple = ("instance", {"affine": True}),
		bias: bool = True,
		dropout: float | tuple = 0.0,
		upsample: str = "deconv",
	) -> None:
		super().__init__()

		img_size = ensure_tuple_rep(img_size, spatial_dims)
		patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
		window_size = ensure_tuple_rep(7, spatial_dims)

		self.normalize = normalize

		self.swinViT = SwinTransformer(
			in_chans=in_channels,
			embed_dim=feature_size,
			window_size=window_size,
			patch_size=patch_sizes,
			depths=depths,
			num_heads=num_heads,
			mlp_ratio=4.0,
			qkv_bias=True,
			drop_rate=drop_rate,
			attn_drop_rate=attn_drop_rate,
			drop_path_rate=dropout_path_rate,
			norm_layer=nn.LayerNorm,
			use_checkpoint=use_checkpoint,
			spatial_dims=spatial_dims,
			downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
			use_v2=use_v2,
		)

		self.conv_0_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=in_channels,out_channels=feature_size,kernel_size=3,stride=1,norm_name=norm_name,res_block=True)
		self.conv_1_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=feature_size,out_channels=2 * feature_size,kernel_size=3,stride=2,norm_name=norm_name, res_block=True)
		self.conv_2_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=4 * feature_size,kernel_size=3,stride=2,norm_name=norm_name,res_block=True)
		self.conv_3_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=4 * feature_size,out_channels=8 * feature_size,kernel_size=3,stride=2,norm_name=norm_name,res_block=True)
		self.conv_4_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=8 * feature_size,out_channels=16 * feature_size,kernel_size=3,stride=2,norm_name=norm_name,res_block=True)
		self.conv_5_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=16 * feature_size,out_channels=32 * feature_size,kernel_size=3,stride=2,norm_name=norm_name,res_block=True)
		self.conv_6_0 = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=32 * feature_size,out_channels=64 * feature_size,kernel_size=3,stride=2,norm_name=norm_name,res_block=True)

		self.upcat_0_1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=1 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_1_1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=4 * feature_size,out_channels=2 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_2_1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=8 * feature_size,out_channels=4 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_3_1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=16 * feature_size,out_channels=8 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_4_1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=32 * feature_size,out_channels=16 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_5_1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=64 * feature_size,out_channels=32 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)

		self.upcat_0_2 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=1 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_1_2 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=4 * feature_size,out_channels=2 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_2_2 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=8 * feature_size,out_channels=4 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_3_2 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=16 * feature_size,out_channels=8 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_4_2 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=32 * feature_size,out_channels=16 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)

		self.upcat_0_3 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=1 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_1_3 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=4 * feature_size,out_channels=2 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_2_3 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=8 * feature_size,out_channels=4 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_3_3 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=16 * feature_size,out_channels=8 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)

		self.upcat_0_4 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=1 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_1_4 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=4 * feature_size,out_channels=2 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_2_4 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=8 * feature_size,out_channels=4 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)

		self.upcat_0_5 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=1 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		self.upcat_1_5 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=4 * feature_size,out_channels=2 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		
		self.upcat_0_6 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=2 * feature_size,out_channels=1 * feature_size,kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
		
		self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
		

	def forward(self, x):
		x_0_0 = self.conv_0_0(x)
		x_1_0 = self.conv_1_0(x_0_0)
		x_0_1 = self.upcat_0_1(x_1_0, x_0_0)

		x_2_0 = self.conv_2_0(x_1_0)
		x_1_1 = self.upcat_1_1(x_2_0, x_1_0)
		x_0_2 = self.upcat_0_2(x_1_1, x_0_1)

		x_3_0 = self.conv_3_0(x_2_0)
		x_2_1 = self.upcat_2_1(x_3_0, x_2_0)
		x_1_2 = self.upcat_1_2(x_2_1, x_1_1)
		x_0_3 = self.upcat_0_3(x_1_2, x_0_2)

		x_4_0 = self.conv_4_0(x_3_0)
		x_3_1 = self.upcat_3_1(x_4_0, x_3_0)
		x_2_2 = self.upcat_2_2(x_3_1, x_2_1)
		x_1_3 = self.upcat_1_3(x_2_2, x_1_2)
		x_0_4 = self.upcat_0_4(x_1_3, x_0_3)	
		
		logits = self.out(x_0_4)

		# print(f"Upsampled tensor shape: {x_3_0.shape}")
		return logits

class ELUNet(nn.Module):

	def __init__(
		self,
		spatial_dims: int = 3,
		in_channels: int = 1,
		out_channels: int = 2,
		features: Sequence[int] = (32, 32, 64, 128, 256, 32),
		deep_supervision: bool = False,
		act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
		norm: str | tuple = ("instance", {"affine": True}),
		bias: bool = True,
		dropout: float | tuple = 0.0,
		upsample: str = "deconv",
	):
		"""
		A UNet++ implementation with 1D/2D/3D supports.

		Based on:

			Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image
			Segmentation". 4th Deep Learning in Medical Image Analysis (DLMIA)
			Workshop, DOI: https://doi.org/10.48550/arXiv.1807.10165


		Args:
			spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
			in_channels: number of input channels. Defaults to 1.
			out_channels: number of output channels. Defaults to 2.
			features: six integers as numbers of features.
				Defaults to ``(32, 32, 64, 128, 256, 32)``,

				- the first five values correspond to the five-level encoder feature sizes.
				- the last value corresponds to the feature size after the last upsampling.

			deep_supervision: whether to prune the network at inference time. Defaults to False. If true, returns a list,
				whose elements correspond to outputs at different nodes.
			act: activation type and arguments. Defaults to LeakyReLU.
			norm: feature normalization type and arguments. Defaults to instance norm.
			bias: whether to have a bias term in convolution blocks. Defaults to True.
				According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
				if a conv layer is directly followed by a batch norm layer, bias should be False.
			dropout: dropout ratio. Defaults to no dropout.
			upsample: upsampling mode, available options are
				``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

		Examples::

			# for spatial 2D
			>>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

			# for spatial 2D, with deep supervision enabled
			>>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), deep_supervision=True)

			# for spatial 2D, with group norm
			>>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

			# for spatial 3D
			>>> net = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

		See Also
			- :py:class:`monai.networks.nets.BasicUNet`
			- :py:class:`monai.networks.nets.DynUNet`
			- :py:class:`monai.networks.nets.UNet`

		"""
		super().__init__()

		fea = ensure_tuple_rep(features, 6)
		print(f"BasicUNetPlusPlus features: {fea}.")
		# Upsamplers for features from level 3 (x_3_0)
		self.upsample_3_to_2 = UpSample(spatial_dims=spatial_dims, in_channels=fea[3], scale_factor=2, mode="pixelshuffle")
		self.upsample_3_to_1 = UpSample(spatial_dims=spatial_dims, in_channels=fea[3], scale_factor=4, mode="pixelshuffle")
		self.upsample_3_to_0 = UpSample(spatial_dims=spatial_dims, in_channels=fea[3], scale_factor=8, mode="pixelshuffle")

		# Upsamplers for features from level 2 (x_2_0)
		self.upsample_2_to_1 = UpSample(spatial_dims=spatial_dims, in_channels=fea[2], scale_factor=2, mode="pixelshuffle")
		self.upsample_2_to_0 = UpSample(spatial_dims=spatial_dims, in_channels=fea[2], scale_factor=4, mode="pixelshuffle")

		# Upsampler for features from level 1 (x_1_0)
		self.upsample_1_to_0 = UpSample(spatial_dims=spatial_dims, in_channels=fea[1], scale_factor=2, mode="pixelshuffle")
		
		self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
		self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
		self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
		self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
		self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

		self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
		self.upcat_2_2 = UpCat(spatial_dims, fea[3], fea[3] + fea[2], fea[2], act, norm, bias, dropout, upsample)
		self.upcat_1_3 = UpCat(spatial_dims, fea[2], fea[3] + fea[2] + fea[1], fea[1], act, norm, bias, dropout, upsample)
		self.upcat_0_4 = UpCat(spatial_dims, fea[1], fea[3] + fea[2] + fea[1] + fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
		
		self.final_conv_0_4 = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


	def forward(self, x: torch.Tensor):
		"""
		Args:
			x: input should have spatially N dimensions
				``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `dimensions`.
				It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
				even edge lengths.

		Returns:
			A torch Tensor of "raw" predictions in shape
			``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
		"""
		x_0_0 = self.conv_0_0(x)
		x_1_0 = self.conv_1_0(x_0_0)
		x_2_0 = self.conv_2_0(x_1_0)
		x_3_0 = self.conv_3_0(x_2_0)
		x_4_0 = self.conv_4_0(x_3_0)
		
		x_3_1 = self.upcat_3_1(x_4_0, x_3_0)
		# For upcat_2_2, target size is x_2_0's size
		skip_2_tensors = [self.upsample_3_to_2(x_3_0), x_2_0]
		x_2_2 = self.upcat_2_2(x_3_1, torch.cat(skip_2_tensors, dim=1))

		# For upcat_1_3, target size is x_1_0's size
		skip_3_tensors = [
			self.upsample_3_to_1(x_3_0),
			self.upsample_2_to_1(x_2_0),
			x_1_0,
		]
		x_1_3 = self.upcat_1_3(x_2_2, torch.cat(skip_3_tensors, dim=1))

		# For upcat_0_4, target size is x_0_0's size
		skip_4_tensors = [
			self.upsample_3_to_0(x_3_0),
			self.upsample_2_to_0(x_2_0),
			self.upsample_1_to_0(x_1_0),
			x_0_0,
		]
		x_0_4 = self.upcat_0_4(x_1_3, torch.cat(skip_4_tensors, dim=1))
		
		output_0_4 = self.final_conv_0_4(x_0_4)

		return output_0_4

class ELUNetPlusPlusPlus(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        deep_supervision: bool = False,
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        """
        A ELU-Net+++ implementation with 1D/2D/3D supports.
        Combines ELU-Net and U-Net3+ architectures with full dense skip connections.

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            deep_supervision: whether to prune the network at inference time. Defaults to False. If true, returns a list,
                whose elements correspond to outputs at different nodes.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = ELUNetPlusPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with deep supervision enabled
            >>> net = ELUNetPlusPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), deep_supervision=True)

            # for spatial 2D, with group norm
            >>> net = ELUNetPlusPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = ELUNetPlusPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
        """
        super().__init__()

        self.deep_supervision = deep_supervision
        self.spatial_dims = spatial_dims

        fea = ensure_tuple_rep(features, 6)
        print(f"ELUNetPlusPlusPlus features: {fea}.")

        # Encoder
        self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # Helper for creating upsampling/downsampling + conv
        def make_inter_scale_conv(in_ch, out_ch, scale_factor):
            """Create inter-scale connection with appropriate up/downsampling"""
            if scale_factor == 1:
                # Same scale
                return Conv["conv", spatial_dims](in_ch, out_ch, kernel_size=3, padding=1)
            elif scale_factor > 1:
                # Upsample
                mode = 'trilinear' if spatial_dims == 3 else 'bilinear'
                return nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                    Conv["conv", spatial_dims](in_ch, out_ch, kernel_size=3, padding=1)
                )
            else:
                # Downsample
                num_pools = int(-scale_factor)
                layers = []
                for _ in range(num_pools):
                    layers.append(nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2))
                layers.append(Conv["conv", spatial_dims](in_ch, out_ch, kernel_size=3, padding=1))
                return nn.Sequential(*layers)

        # ====== Decoder level 3_1 ======
        # Receives from: x_3_0 (same), x_2_0 (down 1x), x_1_0 (down 2x), x_0_0 (down 3x)
        # 4 connections total, each outputs fea[3] channels
        self.conv_3_0_to_3_1 = make_inter_scale_conv(fea[3], fea[3], 1)
        self.conv_2_0_to_3_1 = make_inter_scale_conv(fea[2], fea[3], -1)  # downsample by 2
        self.conv_1_0_to_3_1 = make_inter_scale_conv(fea[1], fea[3], -2)  # downsample by 4
        self.conv_0_0_to_3_1 = make_inter_scale_conv(fea[0], fea[3], -3)  # downsample by 8

        # ====== Decoder level 2_2 ======
        # Receives from: x_2_0 (same), x_1_0 (down 1x), x_0_0 (down 2x), x_3_0 (up 1x), x_4_0 (up 2x)
        # 5 connections total, each outputs fea[2] channels
        self.conv_2_0_to_2_2 = make_inter_scale_conv(fea[2], fea[2], 1)
        self.conv_1_0_to_2_2 = make_inter_scale_conv(fea[1], fea[2], -1)
        self.conv_0_0_to_2_2 = make_inter_scale_conv(fea[0], fea[2], -2)
        self.conv_3_0_to_2_2 = make_inter_scale_conv(fea[3], fea[2], 2)
        self.conv_4_0_to_2_2 = make_inter_scale_conv(fea[4], fea[2], 4)

        # ====== Decoder level 1_3 ======
        # Receives from: x_1_0 (same), x_0_0 (down 1x), x_2_0 (up 1x), x_3_0 (up 2x), x_3_1 (up 2x), x_4_0 (up 3x)
        # 6 connections total, each outputs fea[1] channels
        self.conv_1_0_to_1_3 = make_inter_scale_conv(fea[1], fea[1], 1)
        self.conv_0_0_to_1_3 = make_inter_scale_conv(fea[0], fea[1], -1)
        self.conv_2_0_to_1_3 = make_inter_scale_conv(fea[2], fea[1], 2)
        self.conv_3_0_to_1_3 = make_inter_scale_conv(fea[3], fea[1], 4)
        self.conv_3_1_to_1_3 = make_inter_scale_conv(fea[3], fea[1], 4)
        self.conv_4_0_to_1_3 = make_inter_scale_conv(fea[4], fea[1], 8)

        # ====== Decoder level 0_4 ======
        # Receives from: x_0_0 (same), x_1_0 (up 1x), x_2_0 (up 2x), x_2_2 (up 2x), x_3_0 (up 3x), x_3_1 (up 3x), x_4_0 (up 4x)
        # 7 connections total, each outputs fea[0] channels
        self.conv_0_0_to_0_4 = make_inter_scale_conv(fea[0], fea[0], 1)
        self.conv_1_0_to_0_4 = make_inter_scale_conv(fea[1], fea[0], 2)
        self.conv_2_0_to_0_4 = make_inter_scale_conv(fea[2], fea[0], 4)
        self.conv_2_2_to_0_4 = make_inter_scale_conv(fea[2], fea[0], 4)
        self.conv_3_0_to_0_4 = make_inter_scale_conv(fea[3], fea[0], 8)
        self.conv_3_1_to_0_4 = make_inter_scale_conv(fea[3], fea[0], 8)
        self.conv_4_0_to_0_4 = make_inter_scale_conv(fea[4], fea[0], 16)

        # Decoder UpCat blocks with correct channel counts
        self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3] * 4, fea[3], act, norm, bias, dropout, upsample)
        self.upcat_2_2 = UpCat(spatial_dims, fea[3], fea[2] * 5, fea[2], act, norm, bias, dropout, upsample)
        self.upcat_1_3 = UpCat(spatial_dims, fea[2], fea[1] * 6, fea[1], act, norm, bias, dropout, upsample)
        self.upcat_0_4 = UpCat(spatial_dims, fea[1], fea[0] * 7, fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv_0_4 = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        if deep_supervision:
            self.final_conv_3_1 = Conv["conv", spatial_dims](fea[3], out_channels, kernel_size=1)
            self.final_conv_2_2 = Conv["conv", spatial_dims](fea[2], out_channels, kernel_size=1)
            self.final_conv_1_3 = Conv["conv", spatial_dims](fea[1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        # Encoder
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(x_0_0)
        x_2_0 = self.conv_2_0(x_1_0)
        x_3_0 = self.conv_3_0(x_2_0)
        x_4_0 = self.conv_4_0(x_3_0)

        # Decoder level 3_1: receives from ALL encoder levels (x_0_0, x_1_0, x_2_0, x_3_0)
        skip_3_1 = torch.cat([
            self.conv_3_0_to_3_1(x_3_0),
            self.conv_2_0_to_3_1(x_2_0),
            self.conv_1_0_to_3_1(x_1_0),
            self.conv_0_0_to_3_1(x_0_0)
        ], dim=1)
        x_3_1 = self.upcat_3_1(x_4_0, skip_3_1)

        # Decoder level 2_2: receives from ALL encoder levels (x_0_0 to x_4_0)
        skip_2_2 = torch.cat([
            self.conv_2_0_to_2_2(x_2_0),
            self.conv_1_0_to_2_2(x_1_0),
            self.conv_0_0_to_2_2(x_0_0),
            self.conv_3_0_to_2_2(x_3_0),
            self.conv_4_0_to_2_2(x_4_0)
        ], dim=1)
        x_2_2 = self.upcat_2_2(x_3_1, skip_2_2)

        # Decoder level 1_3: receives from ALL encoders + decoder x_3_1
        skip_1_3 = torch.cat([
            self.conv_1_0_to_1_3(x_1_0),
            self.conv_0_0_to_1_3(x_0_0),
            self.conv_2_0_to_1_3(x_2_0),
            self.conv_3_0_to_1_3(x_3_0),
            self.conv_3_1_to_1_3(x_3_1),
            self.conv_4_0_to_1_3(x_4_0)
        ], dim=1)
        x_1_3 = self.upcat_1_3(x_2_2, skip_1_3)

        # Decoder level 0_4: receives from ALL encoders + decoders x_3_1, x_2_2
        skip_0_4 = torch.cat([
            self.conv_0_0_to_0_4(x_0_0),
            self.conv_1_0_to_0_4(x_1_0),
            self.conv_2_0_to_0_4(x_2_0),
            self.conv_2_2_to_0_4(x_2_2),
            self.conv_3_0_to_0_4(x_3_0),
            self.conv_3_1_to_0_4(x_3_1),
            self.conv_4_0_to_0_4(x_4_0)
        ], dim=1)
        x_0_4 = self.upcat_0_4(x_1_3, skip_0_4)

        # Final output
        output_0_4 = self.final_conv_0_4(x_0_4)

        if self.deep_supervision:
            mode = 'trilinear' if self.spatial_dims == 3 else 'bilinear'
            
            output_3_1 = self.final_conv_3_1(x_3_1)
            output_3_1 = nn.functional.interpolate(output_3_1, size=output_0_4.shape[2:], mode=mode, align_corners=True)
            
            output_2_2 = self.final_conv_2_2(x_2_2)
            output_2_2 = nn.functional.interpolate(output_2_2, size=output_0_4.shape[2:], mode=mode, align_corners=True)
            
            output_1_3 = self.final_conv_1_3(x_1_3)
            output_1_3 = nn.functional.interpolate(output_1_3, size=output_0_4.shape[2:], mode=mode, align_corners=True)
            
            return [output_0_4, output_1_3, output_2_2, output_3_1]

        return output_0_4

