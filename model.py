from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.optim import lr_scheduler
import numpy as np

import monai
from monai.networks.nets import SwinUNETR, BasicUNetPlusPlus
from monai.networks.nets.swin_unetr import *
from monai.networks.layers.factories import Conv
from monai.networks.layers import DropPath, trunc_normal_, Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, UpSample
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep, look_up_option, optional_import, is_sqrt

# Removed UNet, ResNet, Discriminator
class DeepSwinUNETR(nn.Module):

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

		# Encoder blocks refine Swin features (no downsampling, stride=1)
		self.encoder0 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
		self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
		self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=2 * feature_size, out_channels=2 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
		self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=4 * feature_size, out_channels=4 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
		#self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=8 * feature_size, out_channels=8 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)
		#self.encoder5 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=16 * feature_size, out_channels=16 * feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True)

		# Decoder blocks (upsample and merge)
		#self.decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=16 * feature_size, out_channels=8 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
		#self.decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=8 * feature_size, out_channels=4 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
		self.decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=4 * feature_size, out_channels=2 * feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
		self.decoder1 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=2 * feature_size, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
		self.decoder0 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)

		self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

	def forward(self, x):
		# SwinViT extracts hierarchical features (downsampling happens inside)
		# hidden_states[0]: 1x feature_size,  1/2 resolution
		# hidden_states[1]: 2x feature_size,  1/4 resolution
		# hidden_states[2]: 4x feature_size,  1/8 resolution
		# etc.
		hidden_states = self.swinViT(x, self.normalize)

		# Encoder: refine features at each scale
		enc0 = self.encoder0(x)					# Full resolution
		enc1 = self.encoder1(hidden_states[0])	 # 1/2 res
		enc2 = self.encoder2(hidden_states[1])	 # 1/4 res
		enc3 = self.encoder3(hidden_states[2])	 # 1/8 res (bottleneck)
		#enc4 = self.encoder4(hidden_states[3])	 # 1/8 res (bottleneck)
		#enc5 = self.encoder5(hidden_states[4])	 # 1/8 res (bottleneck)

		# Decoder: upsample + skip connections
		#dec4 = self.decoder4(enc5, enc4)		   # 1/8 -> 1/4
		#dec3 = self.decoder3(dec4, enc3)		   # 1/8 -> 1/4
		#dec2 = self.decoder2(dec3, enc2)		   # 1/8 -> 1/4
		dec2 = self.decoder2(enc3, enc2)		   # 1/8 -> 1/4
		dec1 = self.decoder1(dec2, enc1)		   # 1/4 -> 1/2
		dec0 = self.decoder0(dec1, enc0)		   # 1/2 -> full
		return self.out(dec0)

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
