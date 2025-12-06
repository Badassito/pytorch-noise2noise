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

from monai.networks.layers.factories import Conv
from monai.networks.layers import DropPath, trunc_normal_, Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, UpSample
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import, is_sqrt


# Removed UNet, ResNet, Discriminator
class BasicUNetPlusPlus(nn.Module):

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

		self.deep_supervision = deep_supervision

		fea = ensure_tuple_rep(features, 6)
		print(f"BasicUNetPlusPlus features: {fea}.")

		self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
		self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
		self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
		self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
		self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

		self.upcat_0_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)
		self.upcat_1_1 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
		self.upcat_2_1 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
		self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)

		self.upcat_0_2 = UpCat(
			spatial_dims, fea[1], fea[0] * 2, fea[0], act, norm, bias, dropout, upsample, halves=False
		)
		self.upcat_1_2 = UpCat(spatial_dims, fea[2], fea[1] * 2, fea[1], act, norm, bias, dropout, upsample)
		self.upcat_2_2 = UpCat(spatial_dims, fea[3], fea[2] * 2, fea[2], act, norm, bias, dropout, upsample)

		self.upcat_0_3 = UpCat(
			spatial_dims, fea[1], fea[0] * 3, fea[0], act, norm, bias, dropout, upsample, halves=False
		)
		self.upcat_1_3 = UpCat(spatial_dims, fea[2], fea[1] * 3, fea[1], act, norm, bias, dropout, upsample)

		self.upcat_0_4 = UpCat(
			spatial_dims, fea[1], fea[0] * 4, fea[5], act, norm, bias, dropout, upsample, halves=False
		)

		self.final_conv_0_1 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
		self.final_conv_0_2 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
		self.final_conv_0_3 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
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
		x_0_1 = self.upcat_0_1(x_1_0, x_0_0)

		x_2_0 = self.conv_2_0(x_1_0)
		x_1_1 = self.upcat_1_1(x_2_0, x_1_0)
		x_0_2 = self.upcat_0_2(x_1_1, torch.cat([x_0_0, x_0_1], dim=1))

		x_3_0 = self.conv_3_0(x_2_0)
		x_2_1 = self.upcat_2_1(x_3_0, x_2_0)
		x_1_2 = self.upcat_1_2(x_2_1, torch.cat([x_1_0, x_1_1], dim=1))
		x_0_3 = self.upcat_0_3(x_1_2, torch.cat([x_0_0, x_0_1, x_0_2], dim=1))

		x_4_0 = self.conv_4_0(x_3_0)
		x_3_1 = self.upcat_3_1(x_4_0, x_3_0)
		x_2_2 = self.upcat_2_2(x_3_1, torch.cat([x_2_0, x_2_1], dim=1))
		x_1_3 = self.upcat_1_3(x_2_2, torch.cat([x_1_0, x_1_1, x_1_2], dim=1))
		x_0_4 = self.upcat_0_4(x_1_3, torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=1))

		output_0_1 = self.final_conv_0_1(x_0_1)
		output_0_2 = self.final_conv_0_2(x_0_2)
		output_0_3 = self.final_conv_0_3(x_0_3)
		output_0_4 = self.final_conv_0_4(x_0_4)

		if self.deep_supervision:
			output = [output_0_1, output_0_2, output_0_3, output_0_4]
		else:
			output = output_0_4

		return output

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


rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
	"SwinUNETR",
	"window_partition",
	"window_reverse",
	"WindowAttention",
	"SwinTransformerBlock",
	"PatchMerging",
	"PatchMergingV2",
	"MERGING_MODE",
	"BasicLayer",
	"SwinTransformer",
	"SwinUNETRPlusPlus",
	"ELUNet",
	"BasicUNetPlusPlus",
	"ELUNetPlusPlusPlus"
]

class SwinUNETR(nn.Module):
	"""
	Swin UNETR based on: "Hatamizadeh et al.,
	Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
	<https://arxiv.org/abs/2201.01266>"
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		patch_size: int = 2,
		depths: Sequence[int] = (2, 2, 2, 2),
		num_heads: Sequence[int] = (3, 6, 12, 24),
		window_size: Sequence[int] | int = 7,
		qkv_bias: bool = True,
		mlp_ratio: float = 4.0,
		feature_size: int = 24,
		norm_name: tuple | str = "instance",
		drop_rate: float = 0.0,
		attn_drop_rate: float = 0.0,
		dropout_path_rate: float = 0.0,
		normalize: bool = True,
		norm_layer: type[LayerNorm] = nn.LayerNorm,
		patch_norm: bool = False,
		use_checkpoint: bool = False,
		spatial_dims: int = 2,
		downsample: str | nn.Module = "merging",
		use_v2: bool = False,
	) -> None:
		"""
		Args:
			in_channels: dimension of input channels.
			out_channels: dimension of output channels.
			patch_size: size of the patch token.
			feature_size: dimension of network feature size.
			depths: number of layers in each stage.
			num_heads: number of attention heads.
			window_size: local window size.
			qkv_bias: add a learnable bias to query, key, value.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			norm_name: feature normalization type and arguments.
			drop_rate: dropout rate.
			attn_drop_rate: attention dropout rate.
			dropout_path_rate: drop path rate.
			normalize: normalize output intermediate features in each stage.
			norm_layer: normalization layer.
			patch_norm: whether to apply normalization to the patch embedding. Default is False.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
			spatial_dims: number of spatial dims.
			downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
				user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
				The default is currently `"merging"` (the original version defined in v0.9.0).
			use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

		Examples::

			# for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
			>>> net = SwinUNETR(in_channels=1, out_channels=4, feature_size=48)

			# for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
			>>> net = SwinUNETR(in_channels=4, out_channels=3, depths=(2,4,2,2))

			# for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
			>>> net = SwinUNETR(in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

		"""

		super().__init__()

		if spatial_dims not in (2, 3):
			raise ValueError("spatial dimension should be 2 or 3.")

		self.patch_size = patch_size

		patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
		window_size = ensure_tuple_rep(window_size, spatial_dims)

		if not (0 <= drop_rate <= 1):
			raise ValueError("dropout rate should be between 0 and 1.")

		if not (0 <= attn_drop_rate <= 1):
			raise ValueError("attention dropout rate should be between 0 and 1.")

		if not (0 <= dropout_path_rate <= 1):
			raise ValueError("drop path rate should be between 0 and 1.")

		if feature_size % 12 != 0:
			raise ValueError("feature_size should be divisible by 12.")

		self.normalize = normalize

		self.swinViT = SwinTransformer(
			in_chans=in_channels,
			embed_dim=feature_size,
			window_size=window_size,
			patch_size=patch_sizes,
			depths=depths,
			num_heads=num_heads,
			mlp_ratio=mlp_ratio,
			qkv_bias=qkv_bias,
			drop_rate=drop_rate,
			attn_drop_rate=attn_drop_rate,
			drop_path_rate=dropout_path_rate,
			norm_layer=norm_layer,
			patch_norm=patch_norm,
			use_checkpoint=use_checkpoint,
			spatial_dims=spatial_dims,
			downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
			use_v2=use_v2,
		)

		self.encoder1 = UnetrBasicBlock(
			spatial_dims=spatial_dims,
			in_channels=in_channels,
			out_channels=feature_size,
			kernel_size=3,
			stride=1,
			norm_name=norm_name,
			res_block=True,
		)

		self.encoder2 = UnetrBasicBlock(
			spatial_dims=spatial_dims,
			in_channels=feature_size,
			out_channels=feature_size,
			kernel_size=3,
			stride=1,
			norm_name=norm_name,
			res_block=True,
		)

		self.encoder3 = UnetrBasicBlock(
			spatial_dims=spatial_dims,
			in_channels=2 * feature_size,
			out_channels=2 * feature_size,
			kernel_size=3,
			stride=1,
			norm_name=norm_name,
			res_block=True,
		)

		self.encoder4 = UnetrBasicBlock(
			spatial_dims=spatial_dims,
			in_channels=4 * feature_size,
			out_channels=4 * feature_size,
			kernel_size=3,
			stride=1,
			norm_name=norm_name,
			res_block=True,
		)

		self.encoder10 = UnetrBasicBlock(
			spatial_dims=spatial_dims,
			in_channels=16 * feature_size,
			out_channels=16 * feature_size,
			kernel_size=3,
			stride=1,
			norm_name=norm_name,
			res_block=True,
		)

		self.decoder5 = UnetrUpBlock(
			spatial_dims=spatial_dims,
			in_channels=16 * feature_size,
			out_channels=8 * feature_size,
			kernel_size=3,
			upsample_kernel_size=2,
			norm_name=norm_name,
			res_block=True,
		)

		self.decoder4 = UnetrUpBlock(
			spatial_dims=spatial_dims,
			in_channels=feature_size * 8,
			out_channels=feature_size * 4,
			kernel_size=3,
			upsample_kernel_size=2,
			norm_name=norm_name,
			res_block=True,
		)

		self.decoder3 = UnetrUpBlock(
			spatial_dims=spatial_dims,
			in_channels=feature_size * 4,
			out_channels=feature_size * 2,
			kernel_size=3,
			upsample_kernel_size=2,
			norm_name=norm_name,
			res_block=True,
		)
		self.decoder2 = UnetrUpBlock(
			spatial_dims=spatial_dims,
			in_channels=feature_size * 2,
			out_channels=feature_size,
			kernel_size=3,
			upsample_kernel_size=2,
			norm_name=norm_name,
			res_block=True,
		)

		self.decoder1 = UnetrUpBlock(
			spatial_dims=spatial_dims,
			in_channels=feature_size,
			out_channels=feature_size,
			kernel_size=3,
			upsample_kernel_size=2,
			norm_name=norm_name,
			res_block=True,
		)

		self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

	def load_from(self, weights):
		layers1_0: BasicLayer = self.swinViT.layers1[0]  # type: ignore[assignment]
		layers2_0: BasicLayer = self.swinViT.layers2[0]  # type: ignore[assignment]
		layers3_0: BasicLayer = self.swinViT.layers3[0]  # type: ignore[assignment]
		layers4_0: BasicLayer = self.swinViT.layers4[0]  # type: ignore[assignment]
		wstate = weights["state_dict"]

		with torch.no_grad():
			self.swinViT.patch_embed.proj.weight.copy_(wstate["module.patch_embed.proj.weight"])
			self.swinViT.patch_embed.proj.bias.copy_(wstate["module.patch_embed.proj.bias"])
			for bname, block in layers1_0.blocks.named_children():
				block.load_from(weights, n_block=bname, layer="layers1")  # type: ignore[operator]

			if layers1_0.downsample is not None:
				d = layers1_0.downsample
				d.reduction.weight.copy_(wstate["module.layers1.0.downsample.reduction.weight"])  # type: ignore
				d.norm.weight.copy_(wstate["module.layers1.0.downsample.norm.weight"])  # type: ignore
				d.norm.bias.copy_(wstate["module.layers1.0.downsample.norm.bias"])  # type: ignore

			for bname, block in layers2_0.blocks.named_children():
				block.load_from(weights, n_block=bname, layer="layers2")  # type: ignore[operator]

			if layers2_0.downsample is not None:
				d = layers2_0.downsample
				d.reduction.weight.copy_(wstate["module.layers2.0.downsample.reduction.weight"])  # type: ignore
				d.norm.weight.copy_(wstate["module.layers2.0.downsample.norm.weight"])  # type: ignore
				d.norm.bias.copy_(wstate["module.layers2.0.downsample.norm.bias"])  # type: ignore

			for bname, block in layers3_0.blocks.named_children():
				block.load_from(weights, n_block=bname, layer="layers3")  # type: ignore[operator]

			if layers3_0.downsample is not None:
				d = layers3_0.downsample
				d.reduction.weight.copy_(wstate["module.layers3.0.downsample.reduction.weight"])  # type: ignore
				d.norm.weight.copy_(wstate["module.layers3.0.downsample.norm.weight"])  # type: ignore
				d.norm.bias.copy_(wstate["module.layers3.0.downsample.norm.bias"])  # type: ignore

			for bname, block in layers4_0.blocks.named_children():
				block.load_from(weights, n_block=bname, layer="layers4")  # type: ignore[operator]

			if layers4_0.downsample is not None:
				d = layers4_0.downsample
				d.reduction.weight.copy_(wstate["module.layers4.0.downsample.reduction.weight"])  # type: ignore
				d.norm.weight.copy_(wstate["module.layers4.0.downsample.norm.weight"])  # type: ignore
				d.norm.bias.copy_(wstate["module.layers4.0.downsample.norm.bias"])  # type: ignore

	@torch.jit.unused
	def _check_input_size(self, spatial_shape):
		img_size = np.array(spatial_shape)
		remainder = (img_size % np.power(self.patch_size, 5)) > 0
		if remainder.any():
			wrong_dims = (np.where(remainder)[0] + 2).tolist()
			raise ValueError(
				f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
				f" must be divisible by {self.patch_size}**5."
			)

	def forward(self, x_in):
		if not torch.jit.is_scripting() and not torch.jit.is_tracing():
			self._check_input_size(x_in.shape[2:])
		hidden_states_out = self.swinViT(x_in, self.normalize)
		enc0 = self.encoder1(x_in)
		enc1 = self.encoder2(hidden_states_out[0])
		enc2 = self.encoder3(hidden_states_out[1])
		enc3 = self.encoder4(hidden_states_out[2])
		dec4 = self.encoder10(hidden_states_out[4])
		dec3 = self.decoder5(dec4, hidden_states_out[3])
		dec2 = self.decoder4(dec3, enc3)
		dec1 = self.decoder3(dec2, enc2)
		dec0 = self.decoder2(dec1, enc1)
		out = self.decoder1(dec0, enc0)
		logits = self.out(out)
		return logits

def window_partition(x, window_size):
	"""window partition operation based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		x: input tensor.
		window_size: local window size.
	"""
	x_shape = x.size()  # length 4 or 5 only
	if len(x_shape) == 5:
		b, d, h, w, c = x_shape
		x = x.view(
			b,
			d // window_size[0],
			window_size[0],
			h // window_size[1],
			window_size[1],
			w // window_size[2],
			window_size[2],
			c,
		)
		windows = (
			x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
		)
	else:  # if len(x_shape) == 4:
		b, h, w, c = x.shape
		x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
		windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)

	return windows

def window_reverse(windows, window_size, dims):
	"""window reverse operation based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		windows: windows tensor.
		window_size: local window size.
		dims: dimension values.
	"""
	if len(dims) == 4:
		b, d, h, w = dims
		x = windows.view(
			b,
			d // window_size[0],
			h // window_size[1],
			w // window_size[2],
			window_size[0],
			window_size[1],
			window_size[2],
			-1,
		)
		x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

	elif len(dims) == 3:
		b, h, w = dims
		x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
		x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
	return x

def get_window_size(x_size, window_size, shift_size=None):
	"""Computing window size based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		x_size: input size.
		window_size: local window size.
		shift_size: window shifting size.
	"""

	use_window_size = list(window_size)
	if shift_size is not None:
		use_shift_size = list(shift_size)
	for i in range(len(x_size)):
		if x_size[i] <= window_size[i]:
			use_window_size[i] = x_size[i]
			if shift_size is not None:
				use_shift_size[i] = 0

	if shift_size is None:
		return tuple(use_window_size)
	else:
		return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention(nn.Module):
	"""
	Window based multi-head self attention module with relative position bias based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
		self,
		dim: int,
		num_heads: int,
		window_size: Sequence[int],
		qkv_bias: bool = False,
		attn_drop: float = 0.0,
		proj_drop: float = 0.0,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			num_heads: number of attention heads.
			window_size: local window size.
			qkv_bias: add a learnable bias to query, key, value.
			attn_drop: attention dropout rate.
			proj_drop: dropout rate of output.
		"""

		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim**-0.5
		mesh_args = torch.meshgrid.__kwdefaults__

		if len(self.window_size) == 3:
			self.relative_position_bias_table = nn.Parameter(
				torch.zeros(
					(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
					num_heads,
				)
			)
			coords_d = torch.arange(self.window_size[0])
			coords_h = torch.arange(self.window_size[1])
			coords_w = torch.arange(self.window_size[2])
			if mesh_args is not None:
				coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
			else:
				coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
			coords_flatten = torch.flatten(coords, 1)
			relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
			relative_coords = relative_coords.permute(1, 2, 0).contiguous()
			relative_coords[:, :, 0] += self.window_size[0] - 1
			relative_coords[:, :, 1] += self.window_size[1] - 1
			relative_coords[:, :, 2] += self.window_size[2] - 1
			relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
			relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
		elif len(self.window_size) == 2:
			self.relative_position_bias_table = nn.Parameter(
				torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
			)
			coords_h = torch.arange(self.window_size[0])
			coords_w = torch.arange(self.window_size[1])
			if mesh_args is not None:
				coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
			else:
				coords = torch.stack(torch.meshgrid(coords_h, coords_w))
			coords_flatten = torch.flatten(coords, 1)
			relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
			relative_coords = relative_coords.permute(1, 2, 0).contiguous()
			relative_coords[:, :, 0] += self.window_size[0] - 1
			relative_coords[:, :, 1] += self.window_size[1] - 1
			relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

		relative_position_index = relative_coords.sum(-1)
		self.register_buffer("relative_position_index", relative_position_index)
		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)
		trunc_normal_(self.relative_position_bias_table, std=0.02)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, mask):
		b, n, c = x.shape
		qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		q = q * self.scale
		attn = q @ k.transpose(-2, -1)
		relative_position_bias = self.relative_position_bias_table[
			self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore[operator]
		].reshape(n, n, -1)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
		attn = attn + relative_position_bias.unsqueeze(0)
		if mask is not None:
			nw = mask.shape[0]
			attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, n, n)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)

		attn = self.attn_drop(attn).to(v.dtype)
		x = (attn @ v).transpose(1, 2).reshape(b, n, c)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class SwinTransformerBlock(nn.Module):
	"""
	Swin Transformer block based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
		self,
		dim: int,
		num_heads: int,
		window_size: Sequence[int],
		shift_size: Sequence[int],
		mlp_ratio: float = 4.0,
		qkv_bias: bool = True,
		drop: float = 0.0,
		attn_drop: float = 0.0,
		drop_path: float = 0.0,
		act_layer: str = "GELU",
		norm_layer: type[LayerNorm] = nn.LayerNorm,
		use_checkpoint: bool = False,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			num_heads: number of attention heads.
			window_size: local window size.
			shift_size: window shift size.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop: dropout rate.
			attn_drop: attention dropout rate.
			drop_path: stochastic depth rate.
			act_layer: activation layer.
			norm_layer: normalization layer.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
		"""

		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio
		self.use_checkpoint = use_checkpoint
		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention(
			dim,
			window_size=self.window_size,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			attn_drop=attn_drop,
			proj_drop=drop,
		)

		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

	def forward_part1(self, x, mask_matrix):
		x_shape = x.size()
		x = self.norm1(x)
		if len(x_shape) == 5:
			b, d, h, w, c = x.shape
			window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
			pad_l = pad_t = pad_d0 = 0
			pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
			pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
			pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
			x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
			_, dp, hp, wp, _ = x.shape
			dims = [b, dp, hp, wp]

		else:  # elif len(x_shape) == 4
			b, h, w, c = x.shape
			window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
			pad_l = pad_t = 0
			pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
			pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
			x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
			_, hp, wp, _ = x.shape
			dims = [b, hp, wp]

		if any(i > 0 for i in shift_size):
			if len(x_shape) == 5:
				shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
			elif len(x_shape) == 4:
				shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
			attn_mask = mask_matrix
		else:
			shifted_x = x
			attn_mask = None
		x_windows = window_partition(shifted_x, window_size)
		attn_windows = self.attn(x_windows, mask=attn_mask)
		attn_windows = attn_windows.view(-1, *(window_size + (c,)))
		shifted_x = window_reverse(attn_windows, window_size, dims)
		if any(i > 0 for i in shift_size):
			if len(x_shape) == 5:
				x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
			elif len(x_shape) == 4:
				x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
		else:
			x = shifted_x

		if len(x_shape) == 5:
			if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
				x = x[:, :d, :h, :w, :].contiguous()
		elif len(x_shape) == 4:
			if pad_r > 0 or pad_b > 0:
				x = x[:, :h, :w, :].contiguous()

		return x

	def forward_part2(self, x):
		return self.drop_path(self.mlp(self.norm2(x)))

	def load_from(self, weights, n_block, layer):
		root = f"module.{layer}.0.blocks.{n_block}."
		block_names = [
			"norm1.weight",
			"norm1.bias",
			"attn.relative_position_bias_table",
			"attn.relative_position_index",
			"attn.qkv.weight",
			"attn.qkv.bias",
			"attn.proj.weight",
			"attn.proj.bias",
			"norm2.weight",
			"norm2.bias",
			"mlp.fc1.weight",
			"mlp.fc1.bias",
			"mlp.fc2.weight",
			"mlp.fc2.bias",
		]
		with torch.no_grad():
			self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
			self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
			self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
			self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore[operator]
			self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
			self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
			self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
			self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
			self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
			self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
			self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
			self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
			self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
			self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

	def forward(self, x, mask_matrix):
		shortcut = x
		if self.use_checkpoint:
			x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
		else:
			x = self.forward_part1(x, mask_matrix)
		x = shortcut + self.drop_path(x)
		if self.use_checkpoint:
			x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
		else:
			x = x + self.forward_part2(x)
		return x

class PatchMergingV2(nn.Module):
	"""
	Patch merging layer based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
		"""
		Args:
			dim: number of feature channels.
			norm_layer: normalization layer.
			spatial_dims: number of spatial dims.
		"""

		super().__init__()
		self.dim = dim
		if spatial_dims == 3:
			self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
			self.norm = norm_layer(8 * dim)
		elif spatial_dims == 2:
			self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
			self.norm = norm_layer(4 * dim)

	def forward(self, x):
		x_shape = x.size()
		if len(x_shape) == 5:
			b, d, h, w, c = x_shape
			pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
			if pad_input:
				x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
			x = torch.cat(
				[x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
			)

		elif len(x_shape) == 4:
			b, h, w, c = x_shape
			pad_input = (h % 2 == 1) or (w % 2 == 1)
			if pad_input:
				x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
			x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

		x = self.norm(x)
		x = self.reduction(x)
		return x

class PatchMerging(PatchMergingV2):
	"""The `PatchMerging` module previously defined in v0.9.0."""

	def forward(self, x):
		x_shape = x.size()
		if len(x_shape) == 4:
			return super().forward(x)
		if len(x_shape) != 5:
			raise ValueError(f"expecting 5D x, got {x.shape}.")
		b, d, h, w, c = x_shape
		pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
		if pad_input:
			x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
		x0 = x[:, 0::2, 0::2, 0::2, :]
		x1 = x[:, 1::2, 0::2, 0::2, :]
		x2 = x[:, 0::2, 1::2, 0::2, :]
		x3 = x[:, 0::2, 0::2, 1::2, :]
		x4 = x[:, 1::2, 1::2, 0::2, :]
		x5 = x[:, 1::2, 0::2, 1::2, :]
		x6 = x[:, 0::2, 1::2, 1::2, :]
		x7 = x[:, 1::2, 1::2, 1::2, :]
		x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
		x = self.norm(x)
		x = self.reduction(x)
		return x

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

def compute_mask(dims, window_size, shift_size, device):
	"""Computing region masks based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		dims: dimension values.
		window_size: local window size.
		shift_size: shift size.
		device: device.
	"""

	cnt = 0

	if len(dims) == 3:
		d, h, w = dims
		img_mask = torch.zeros((1, d, h, w, 1), device=device)
		for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
			for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
				for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
					img_mask[:, d, h, w, :] = cnt
					cnt += 1

	elif len(dims) == 2:
		h, w = dims
		img_mask = torch.zeros((1, h, w, 1), device=device)
		for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
			for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
				img_mask[:, h, w, :] = cnt
				cnt += 1

	mask_windows = window_partition(img_mask, window_size)
	mask_windows = mask_windows.squeeze(-1)
	attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
	attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

	return attn_mask

class BasicLayer(nn.Module):
	"""
	Basic Swin Transformer layer in one stage based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
		self,
		dim: int,
		depth: int,
		num_heads: int,
		window_size: Sequence[int],
		drop_path: list,
		mlp_ratio: float = 4.0,
		qkv_bias: bool = False,
		drop: float = 0.0,
		attn_drop: float = 0.0,
		norm_layer: type[LayerNorm] = nn.LayerNorm,
		downsample: nn.Module | None = None,
		use_checkpoint: bool = False,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			depth: number of layers in each stage.
			num_heads: number of attention heads.
			window_size: local window size.
			drop_path: stochastic depth rate.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop: dropout rate.
			attn_drop: attention dropout rate.
			norm_layer: normalization layer.
			downsample: an optional downsampling layer at the end of the layer.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
		"""

		super().__init__()
		self.window_size = window_size
		self.shift_size = tuple(i // 2 for i in window_size)
		self.no_shift = tuple(0 for i in window_size)
		self.depth = depth
		self.use_checkpoint = use_checkpoint
		self.blocks = nn.ModuleList(
			[
				SwinTransformerBlock(
					dim=dim,
					num_heads=num_heads,
					window_size=self.window_size,
					shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					drop=drop,
					attn_drop=attn_drop,
					drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
					norm_layer=norm_layer,
					use_checkpoint=use_checkpoint,
				)
				for i in range(depth)
			]
		)
		self.downsample = downsample
		if callable(self.downsample):
			self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

	def forward(self, x):
		x_shape = x.size()
		if len(x_shape) == 5:
			b, c, d, h, w = x_shape
			window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
			x = rearrange(x, "b c d h w -> b d h w c")
			dp = int(np.ceil(d / window_size[0])) * window_size[0]
			hp = int(np.ceil(h / window_size[1])) * window_size[1]
			wp = int(np.ceil(w / window_size[2])) * window_size[2]
			attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
			for blk in self.blocks:
				x = blk(x, attn_mask)
			x = x.view(b, d, h, w, -1)
			if self.downsample is not None:
				x = self.downsample(x)
			x = rearrange(x, "b d h w c -> b c d h w")

		elif len(x_shape) == 4:
			b, c, h, w = x_shape
			window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
			x = rearrange(x, "b c h w -> b h w c")
			hp = int(np.ceil(h / window_size[0])) * window_size[0]
			wp = int(np.ceil(w / window_size[1])) * window_size[1]
			attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
			for blk in self.blocks:
				x = blk(x, attn_mask)
			x = x.view(b, h, w, -1)
			if self.downsample is not None:
				x = self.downsample(x)
			x = rearrange(x, "b h w c -> b c h w")
		return x

class SwinTransformer(nn.Module):
	"""
	Swin Transformer based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
		self,
		in_chans: int,
		embed_dim: int,
		window_size: Sequence[int],
		patch_size: Sequence[int],
		depths: Sequence[int],
		num_heads: Sequence[int],
		mlp_ratio: float = 4.0,
		qkv_bias: bool = True,
		drop_rate: float = 0.0,
		attn_drop_rate: float = 0.0,
		drop_path_rate: float = 0.0,
		norm_layer: type[LayerNorm] = nn.LayerNorm,
		patch_norm: bool = False,
		use_checkpoint: bool = False,
		spatial_dims: int = 3,
		downsample="merging",
		use_v2=False,
	) -> None:
		"""
		Args:
			in_chans: dimension of input channels.
			embed_dim: number of linear projection output channels.
			window_size: local window size.
			patch_size: patch size.
			depths: number of layers in each stage.
			num_heads: number of attention heads.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop_rate: dropout rate.
			attn_drop_rate: attention dropout rate.
			drop_path_rate: stochastic depth rate.
			norm_layer: normalization layer.
			patch_norm: add normalization after patch embedding.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
			spatial_dims: spatial dimension.
			downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
				user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
				The default is currently `"merging"` (the original version defined in v0.9.0).
			use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
		"""

		super().__init__()
		self.num_layers = len(depths)
		self.embed_dim = embed_dim
		self.patch_norm = patch_norm
		self.window_size = window_size
		self.patch_size = patch_size
		self.patch_embed = PatchEmbed(
			patch_size=self.patch_size,
			in_chans=in_chans,
			embed_dim=embed_dim,
			norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
			spatial_dims=spatial_dims,
		)
		self.pos_drop = nn.Dropout(p=drop_rate)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
		self.use_v2 = use_v2
		self.layers1 = nn.ModuleList()
		self.layers2 = nn.ModuleList()
		self.layers3 = nn.ModuleList()
		self.layers4 = nn.ModuleList()
		if self.use_v2:
			self.layers1c = nn.ModuleList()
			self.layers2c = nn.ModuleList()
			self.layers3c = nn.ModuleList()
			self.layers4c = nn.ModuleList()
		down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
		for i_layer in range(self.num_layers):
			layer = BasicLayer(
				dim=int(embed_dim * 2**i_layer),
				depth=depths[i_layer],
				num_heads=num_heads[i_layer],
				window_size=self.window_size,
				drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
				mlp_ratio=mlp_ratio,
				qkv_bias=qkv_bias,
				drop=drop_rate,
				attn_drop=attn_drop_rate,
				norm_layer=norm_layer,
				downsample=down_sample_mod,
				use_checkpoint=use_checkpoint,
			)
			if i_layer == 0:
				self.layers1.append(layer)
			elif i_layer == 1:
				self.layers2.append(layer)
			elif i_layer == 2:
				self.layers3.append(layer)
			elif i_layer == 3:
				self.layers4.append(layer)
			if self.use_v2:
				layerc = UnetrBasicBlock(
					spatial_dims=spatial_dims,
					in_channels=embed_dim * 2**i_layer,
					out_channels=embed_dim * 2**i_layer,
					kernel_size=3,
					stride=1,
					norm_name="instance",
					res_block=True,
				)
				if i_layer == 0:
					self.layers1c.append(layerc)
				elif i_layer == 1:
					self.layers2c.append(layerc)
				elif i_layer == 2:
					self.layers3c.append(layerc)
				elif i_layer == 3:
					self.layers4c.append(layerc)

		self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

	def proj_out(self, x, normalize=False):
		if normalize:
			x_shape = x.shape
			# Force trace() to generate a constant by casting to int
			ch = int(x_shape[1])
			if len(x_shape) == 5:
				x = rearrange(x, "n c d h w -> n d h w c")
				x = F.layer_norm(x, [ch])
				x = rearrange(x, "n d h w c -> n c d h w")
			elif len(x_shape) == 4:
				x = rearrange(x, "n c h w -> n h w c")
				x = F.layer_norm(x, [ch])
				x = rearrange(x, "n h w c -> n c h w")
		return x

	def forward(self, x, normalize=True):
		x0 = self.patch_embed(x)
		x0 = self.pos_drop(x0)
		x0_out = self.proj_out(x0, normalize)
		if self.use_v2:
			x0 = self.layers1c[0](x0.contiguous())
		x1 = self.layers1[0](x0.contiguous())
		x1_out = self.proj_out(x1, normalize)
		if self.use_v2:
			x1 = self.layers2c[0](x1.contiguous())
		x2 = self.layers2[0](x1.contiguous())
		x2_out = self.proj_out(x2, normalize)
		if self.use_v2:
			x2 = self.layers3c[0](x2.contiguous())
		x3 = self.layers3[0](x2.contiguous())
		x3_out = self.proj_out(x3, normalize)
		if self.use_v2:
			x3 = self.layers4c[0](x3.contiguous())
		x4 = self.layers4[0](x3.contiguous())
		x4_out = self.proj_out(x4, normalize)
		return [x0_out, x1_out, x2_out, x3_out, x4_out]

def filter_swinunetr(key, value):
	"""
	A filter function used to filter the pretrained weights from [1], then the weights can be loaded into MONAI SwinUNETR Model.
	This function is typically used with `monai.networks.copy_model_state`
	[1] "Valanarasu JM et al., Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training
	<https://arxiv.org/abs/2307.16896>"

	Args:
		key: the key in the source state dict used for the update.
		value: the value in the source state dict used for the update.

	Examples::

		import torch
		from monai.apps import download_url
		from monai.networks.utils import copy_model_state
		from monai.networks.nets.swin_unetr import SwinUNETR, filter_swinunetr

		model = SwinUNETR(in_channels=1, out_channels=3, feature_size=48)
		resource = (
			"https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
		)
		ssl_weights_path = "./ssl_pretrained_weights.pth"
		download_url(resource, ssl_weights_path)
		ssl_weights = torch.load(ssl_weights_path, weights_only=True)["model"]

		dst_dict, loaded, not_loaded = copy_model_state(model, ssl_weights, filter_func=filter_swinunetr)

	"""
	if key in [
		"encoder.mask_token",
		"encoder.norm.weight",
		"encoder.norm.bias",
		"out.conv.conv.weight",
		"out.conv.conv.bias",
	]:
		return None

	if key[:8] == "encoder.":
		if key[8:19] == "patch_embed":
			new_key = "swinViT." + key[8:]
		else:
			new_key = "swinViT." + key[8:18] + key[20:]

		return new_key, value
	else:
		return None

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
