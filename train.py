from model import *
from model import init_net
from dataset import *

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import monai
import monai.transforms as monai_transforms
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image import RelativeAverageSpectralError
from torchmetrics.image import PeakSignalNoiseRatio
import albumentations as A

##
class Train:
	def __init__(self, args):
		self.mode = args.mode
		self.train_continue = args.train_continue

		self.scope = args.scope
		self.norm = args.norm

		self.dir_checkpoint = args.dir_checkpoint
		self.dir_log = args.dir_log

		self.name_data = args.name_data
		self.dir_data = args.dir_data
		self.dir_result = args.dir_result

		self.num_epoch = args.num_epoch
		self.batch_size = args.batch_size

		self.lr_G = args.lr_G

		self.optim = args.optim
		self.beta1 = args.beta1

		self.ny_load = args.ny_load
		self.nx_load = args.nx_load
		self.nch_load = args.nch_load

		self.ny_out = args.ny_out
		self.nx_out = args.nx_out
		self.nch_out = args.nch_out

		self.data_type = args.data_type

		self.num_freq_disp = args.num_freq_disp
		self.num_freq_save = args.num_freq_save

		# Performance optimization flags
		self.use_amp = getattr(args, 'use_amp', True)
		self.use_compile = getattr(args, 'use_compile', False)
		self.use_grad_checkpoint = getattr(args, 'use_grad_checkpoint', True)
		self.use_scheduler = getattr(args, 'use_scheduler', True)
		self.save_images = getattr(args, 'save_images', False)

		self.gpu_ids = args.gpu_ids

		if self.gpu_ids and torch.cuda.is_available():
			self.device = torch.device("cuda:%d" % self.gpu_ids[0])
			torch.cuda.set_device(self.gpu_ids[0])
		else:
			self.device = torch.device("cpu")

	def save(self, dir_chck, netG, optimG, epoch):
		if not os.path.exists(dir_chck):
			os.makedirs(dir_chck)

		torch.save({'netG': netG.state_dict(),
					'optimG': optimG.state_dict()},
				   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

	def load(self, dir_chck, netG, optimG=[], epoch=[], mode='train'):

		if not os.path.exists(dir_chck):
			epoch = 0
			if mode == 'train':
				return netG, optimG, epoch
			elif mode == 'test':
				return netG, epoch

		if not epoch:
			ckpt = os.listdir(dir_chck)
			ckpt.sort()
			epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

		dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

		print('Loaded %dth network' % epoch)

		if mode == 'train':
			netG.load_state_dict(dict_net['netG'])
			optimG.load_state_dict(dict_net['optimG'])

			return netG, optimG, epoch

		elif mode == 'test':
			netG.load_state_dict(dict_net['netG'])

			return netG, epoch

	def train(self):
		mode = self.mode

		train_continue = self.train_continue
		num_epoch = self.num_epoch

		lr_G = self.lr_G

		batch_size = self.batch_size
		device = self.device

		gpu_ids = self.gpu_ids

		nch_out = self.nch_out

		norm = self.norm
		name_data = self.name_data

		num_freq_disp = self.num_freq_disp
		num_freq_save = self.num_freq_save

		## setup dataset
		dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

		dir_data_train = os.path.join(self.dir_data, name_data, 'train')
		dir_data_val = os.path.join(self.dir_data, name_data, 'val')

		dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
		dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')

		dir_result_train = os.path.join(self.dir_result, self.scope, name_data, 'train')
		dir_result_val = os.path.join(self.dir_result, self.scope, name_data, 'val')
		if not os.path.exists(os.path.join(dir_result_train, 'images')):
			os.makedirs(os.path.join(dir_result_train, 'images'))
		if not os.path.exists(os.path.join(dir_result_val, 'images')):
			os.makedirs(os.path.join(dir_result_val, 'images'))

		transform_train = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((self.ny_load, self.nx_load)), ToTensor()])
		transform_val = transforms.Compose([Normalize(mean=0.5, std=0.5), RandomFlip(), RandomCrop((self.ny_load, self.nx_load)), ToTensor()])

		# Albumentations transforms for training and validation
		
	# 	transform_train = A.Compose([
	# 		A.D4(p=1.0), #https://explore.albumentations.ai/transform/D4
			# A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), p=0.5), #https://explore.albumentations.ai/transform/ColorJitter
			# A.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-30, 30), shear={"x": (-10, 10), "y": (-10, 10)}, fill=0, fit_output=False, keep_ratio=False, balanced_scale=True, p=0.5), #https://explore.albumentations.ai/transform/Affine
	# 		A.RandomCrop(width=self.nx_load, height=self.ny_load),
			# A.AdditiveNoise(noise_type="gaussian", spatial_mode="per_pixel", approximation=1.0, noise_params={"mean_range": (0.0, 0.0),"std_range": (0.5, 0.5) }, p=1), # https://explore.albumentations.ai/transform/AdditiveNoise
			# A.RandomToneCurve(scale=0.2, per_channel=True, p=0.5),# https://explore.albumentations.ai/transform/RandomToneCurve
			# A.ShotNoise(scale_range=(0.1, 10.0),p=0.5),#https://explore.albumentations.ai/transform/ShotNoise
			# A.RandomGamma(gamma_limit=(80, 120), p=0.5), #https://explore.albumentations.ai/transform/RandomGamma
			# A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, ensure_safe_range=False, p=0.5), # https://explore.albumentations.ai/transform/RandomBrightnessContrast
			# A.ElasticTransform(alpha=1.0, sigma=50.0, approximate=False, same_dxdy=False, noise_distribution="gaussian", p=0.5), #https://explore.albumentations.ai/transform/ElasticTransform
			# A.Perspective(scale=(0.05, 0.1), keep_size=True, fill=0, fill_mask=0, fit_output=False, p=0.5), #https://explore.albumentations.ai/transform/Perspective
			# A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), normalized=True, p=0.5), #https://explore.albumentations.ai/transform/GridDistortion
	# 	])

	# 	transform_val = A.Compose([
	# 		A.RandomCrop(width=self.nx_load, height=self.ny_load),
			# A.HorizontalFlip(p=0.5),
			# A.VerticalFlip(p=0.5),
	# 	])

		transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])

		dataset_train = Dataset(dir_data_train, data_type=self.data_type, transform=transform_train, sgm=(26, 26))
		dataset_val = Dataset(dir_data_val, data_type=self.data_type, transform=transform_val, sgm=(26, 26))

		loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
		loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

		num_train = len(dataset_train)
		num_val = len(dataset_val)

		num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
		num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

		## setup network
		# netG = BasicUNetPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, features=(64,64,128,256,512,64))
		# netG = ELUNetPlusPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, features=(32,32,64,128,256,32))
		# netG = ELUNet(spatial_dims=2, in_channels=1, out_channels=1, features=(32,32,64,128,256,32))
		# Enable gradient checkpointing for memory efficiency if requested
		use_checkpoint = self.use_grad_checkpoint
		netG = SwinUNETR(spatial_dims=2, in_channels=1, out_channels=1, depths=(3,3,3,3), feature_size=24, use_v2=True, use_checkpoint=use_checkpoint)
		# netG = SwinUNETRPlusPlus(img_size=(self.ny_load, self.nx_load), spatial_dims=2, in_channels=1, out_channels=1, depths=(3,3,3,3), feature_size=24,num_heads=(6,12,24,48), use_v2=True, drop_rate=0.2, attn_drop_rate=0.2, dropout_path_rate=0.2, downsample="mergingv2")
		netG = netG.to(memory_format=torch.channels_last)

		init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

		# Apply torch.compile for faster training (PyTorch 2.0+)
		if self.use_compile:
			print("Compiling model with torch.compile()...")
			netG = torch.compile(netG, mode='reduce-overhead')

		## setup loss & optimization
		# fn_REG = nn.L1Loss().to(device)  # Regression loss: L1
		# fn_REG1 = nn.MSELoss().to(device)	 # Regression loss: L2
		# fn_REG1 = DeepImageStructureAndTextureSimilarity().to(device)
		fn_REG = RootMeanSquaredErrorUsingSlidingWindow().to(device)
		# fn_REG2 = PeakSignalNoiseRatio(data_range=1.0).to(device)
		paramsG = netG.parameters()

		optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999), fused=True)

		# Setup GradScaler for mixed precision training
		scaler = GradScaler('cuda', enabled=self.use_amp)
		if self.use_amp:
			print("Mixed precision training (AMP) enabled")

		# Setup learning rate scheduler
		schedG = None
		if self.use_scheduler:
			schedG = CosineAnnealingLR(optimG, T_max=num_epoch, eta_min=lr_G * 0.01)
			print(f"Cosine annealing scheduler enabled (T_max={num_epoch}, eta_min={lr_G * 0.01})")

		## load from checkpoints
		st_epoch = 0

		if train_continue == 'on':
			netG, optimG, st_epoch = self.load(dir_chck, netG, optimG, mode=mode)

		## setup tensorboard
		# DISABLED LOGGING
		# writer_train = SummaryWriter(log_dir=dir_log_train)
		# writer_val = SummaryWriter(log_dir=dir_log_val)

		for epoch in range(st_epoch + 1, num_epoch + 1):
			## training phase
			netG.train()

			loss_G_train = []

			for batch, data in enumerate(loader_train, 1):
				def should(freq):
					return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

				label = data['label'].to(device, non_blocking=True, memory_format=torch.channels_last)
				input = data['input'].to(device, non_blocking=True, memory_format=torch.channels_last)

				# backward netG with mixed precision
				optimG.zero_grad()
				with autocast('cuda', enabled=self.use_amp):
					output = netG(input)
				# Compute loss outside autocast (torchmetrics doesn't support mixed dtypes)
				loss_G = fn_REG(output.float(), label)

				scaler.scale(loss_G).backward()
				scaler.step(optimG)
				scaler.update()

				# get losses
				loss_G_train += [loss_G.item()]

				print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
					  % (epoch, batch, num_batch_train, np.mean(loss_G_train)))

				# Only save images if explicitly requested (disabled by default for speed)
				if self.save_images and should(num_freq_disp):
					## show output
					input = transform_inv(input)
					label = transform_inv(label)
					output = transform_inv(output)

					input = np.clip(input, 0, 1)
					label = np.clip(label, 0, 1)
					output = np.clip(output, 0, 1)
					#DISABLED LOGGING
					# writer_train.add_images('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
					# writer_train.add_images('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
					# writer_train.add_images('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

					for j in range(label.shape[0]):
						# name = num_train * (epoch - 1) + num_batch_train * (batch - 1) + j
						name = num_batch_train * (batch - 1) + j
						fileset = {'name': name,
								   'input': "%04d-input.png" % name,
								   'output': "%04d-output.png" % name,
								   'label': "%04d-label.png" % name}

						Image.fromarray((input[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_train,'images', fileset['input']))
						Image.fromarray((output[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_train,'images', fileset['output']))
						Image.fromarray((label[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_train,'images', fileset['label']))
						# DISABLED LOGGING
						# append_index(dir_result_train, fileset)
			# DISABLED LOGGING
			# writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)

			## validation phase
			with torch.no_grad():
				netG.eval()

				loss_G_val = []

				for batch, data in enumerate(loader_val, 1):
					def should(freq):
						return freq > 0 and (batch % freq == 0 or batch == num_batch_val)

					label = data['label'].to(device, non_blocking=True, memory_format=torch.channels_last)
					input = data['input'].to(device, non_blocking=True, memory_format=torch.channels_last)

					# forward netG with mixed precision
					with autocast('cuda', enabled=self.use_amp):
						output = netG(input)
					# Compute loss outside autocast (torchmetrics doesn't support mixed dtypes)
					loss_G = fn_REG(output.float(), label)

					loss_G_val += [loss_G.item()]

					print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
						  % (epoch, batch, num_batch_val, np.mean(loss_G_val)))

					# Only save images if explicitly requested (disabled by default for speed)
					if self.save_images and should(num_freq_disp):
						## show output
						input = transform_inv(input)
						label = transform_inv(label)
						output = transform_inv(output)

						input = np.clip(input, 0, 1)
						label = np.clip(label, 0, 1)
						output = np.clip(output, 0, 1)
						# DISABLED LOGGING
						# writer_val.add_images('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
						# writer_val.add_images('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
						# writer_val.add_images('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

						for j in range(label.shape[0]):
							# name = num_train * (epoch - 1) + num_batch_train * (batch - 1) + j
							name = num_batch_train * (batch - 1) + j
							fileset = {'name': name,
									   'input': "%04d-input.png" % name,
									   'output': "%04d-output.png" % name,
									   'label': "%04d-label.png" % name}

							Image.fromarray((input[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_val, 'images', fileset['input']))
							Image.fromarray((output[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_val, 'images', fileset['output']))
							Image.fromarray((label[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_val, 'images', fileset['label']))
							# DISABLED LOGGING
							# append_index(dir_result_val, fileset)
				# DISABLED LOGGING
				# writer_val.add_scalar('loss_G', np.mean(loss_G_val), epoch)

			# Update learning rate scheduler
			if schedG is not None:
				schedG.step()

			## save
			if (epoch % num_freq_save) == 0:
				self.save(dir_chck, netG, optimG, epoch)
		# DISABLED LOGGING
		# writer_train.close()
		# writer_val.close()

	def test(self):
		"""Modified test function using MONAI SlidingWindowInferer"""
		from monai.inferers import SlidingWindowInferer
		
		mode = self.mode
		batch_size = self.batch_size
		device = self.device
		gpu_ids = self.gpu_ids

		nch_out = self.nch_out
		norm = self.norm
		name_data = self.name_data

		## setup dataset
		dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)
		dir_result_test = os.path.join(self.dir_result, self.scope, name_data, 'test')
		if not os.path.exists(os.path.join(dir_result_test, 'images')):
			os.makedirs(os.path.join(dir_result_test, 'images'))

		dir_data_test = os.path.join(self.dir_data, name_data, 'test')

		# No spatial transforms for test - just process full images with sliding window inference
		transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])

		# Create dataset without any augmentations for test mode
		class FullImageDataset(torch.utils.data.Dataset):
			def __init__(self, data_dir, sgm=(0, 26)):
				self.data_dir = data_dir
				self.sgm_input = sgm[1]

				lst_data = os.listdir(data_dir)
				lst_data.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
				self.lst_data = lst_data

			def __getitem__(self, index):
				data = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))

				if data.dtype == np.uint8:
					data = data / 255.0
				if data.ndim == 2:
					data = np.expand_dims(data, axis=2)
				# if data.shape[0] > data.shape[1]:
				# 	data = data.transpose((1, 0, 2))

				sz = data.shape
				# Only add noise to input, keep clean label
				label = data.copy()
				input = data + self.sgm_input/255 * np.random.randn(sz[0], sz[1], sz[2])

				# Convert to tensor and normalize (no spatial transforms for test)
				input = torch.from_numpy(input.transpose((2, 0, 1)).astype(np.float32))
				label = torch.from_numpy(label.transpose((2, 0, 1)).astype(np.float32))

				# Apply normalization
				input = (input - 0.5) / 0.5
				label = (label - 0.5) / 0.5

				return {'input': input, 'label': label}

			def __len__(self):
				return len(self.lst_data)

		dataset_test = FullImageDataset(dir_data_test, sgm=(0, 26))
		loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)

		## setup network
		# netG = ELUNet(spatial_dims=2, in_channels=1, out_channels=1, features=(64,64,128,256,512,64))
		# netG = ELUNetPlusPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, features=(32,32,64,128,256,32))
		# netG = BasicUNetPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, features=(64,64,128,256,512,64))
		netG = SwinUNETR(spatial_dims=2, in_channels=1, out_channels=1, depths=(3,3,3,3), feature_size=24, use_v2=True)
		# netG = SwinUNETRPlusPlus(img_size=(self.ny_load, self.nx_load), spatial_dims=2, in_channels=1, out_channels=1, depths=(3,3,3,3), feature_size=24,num_heads=(6,12,24,48), use_v2=True, drop_rate=0.2, attn_drop_rate=0.2, dropout_path_rate=0.2, downsample="mergingv2")
		netG = netG.to(memory_format=torch.channels_last)
		init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

		## load from checkpoints
		netG, st_epoch = self.load(dir_chck, netG, mode=mode)

		# Setup SlidingWindowInferer
		inferer = SlidingWindowInferer(
			roi_size=(self.ny_load, self.nx_load), # Use the training patch size
			sw_batch_size=batch_size,  # Process multiple patches at once
			overlap=0.75,	 # 25% overlap between patches
			mode='gaussian',  # Use gaussian blending for smoother results
			padding_mode='reflect'  # Padding for edge cases
		)

		## setup loss function
		# fn_REG = nn.L1Loss().to(device)
		# fn_REG = ARNIQA().to(device)
		fn_REG = RootMeanSquaredErrorUsingSlidingWindow().to(device)
		# fn_REG = fn_REG1 + fn_REG2

		## test phase
		with torch.no_grad():
			netG.eval()
			loss_G_test = []

			for i, data in enumerate(loader_test, 1):
				label = data['label'].to(device, non_blocking=True, memory_format=torch.channels_last)
				input = data['input'].to(device, non_blocking=True, memory_format=torch.channels_last)

				# Use sliding window inference with mixed precision
				with autocast('cuda', enabled=self.use_amp):
					output = inferer(input, netG)
				# Compute loss outside autocast (torchmetrics doesn't support mixed dtypes)
				loss_G = fn_REG(output.float(), label)

				loss_G_test += [loss_G.item()]

				# Convert back to numpy for saving
				input = transform_inv(input)
				label = transform_inv(label)  
				output = transform_inv(output)

				input = np.clip(input, 0, 1)
				label = np.clip(label, 0, 1)
				output = np.clip(output, 0, 1)

				# Save results
				for j in range(label.shape[0]):
					name = i - 1 + j  # Since batch_size=1, this is just i-1
					fileset = {
						'name': name,
						'input': "%04d-input.png" % name,
						'output': "%04d-output.png" % name,
						'label': "%04d-label.png" % name
					}

					Image.fromarray((input[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_test, 'images', fileset['input']))
					Image.fromarray((output[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_test, 'images', fileset['output']))
					Image.fromarray((label[j, :, :, :].squeeze() * 255).astype(np.uint8)).save(os.path.join(dir_result_test, 'images', fileset['label']))
					# DISABLED LOGGING
					# append_index(dir_result_test, fileset)

				print('TEST: %d/%d: LOSS: %.6f' % (i, len(loader_test), loss_G.item()))
				
			print('TEST: AVERAGE LOSS: %.6f' % (np.mean(loss_G_test)))


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
	"""Return a learning rate scheduler

	Parameters:
		optimizer		  -- the optimizer of the network
		opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
							  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

	For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
	and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
	For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
	See https://pytorch.org/docs/stable/optim.html for more details.
	"""
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def append_index(dir_result, fileset, step=False):
	index_path = os.path.join(dir_result, "index.html")
	if os.path.exists(index_path):
		index = open(index_path, "a")
	else:
		index = open(index_path, "w")
		index.write("<html><body><table><tr>")
		if step:
			index.write("<th>step</th>")
		for key, value in fileset.items():
			index.write("<th>%s</th>" % key)
		index.write('</tr>')

	# for fileset in filesets:
	index.write("<tr>")

	if step:
		index.write("<td>%d</td>" % fileset["step"])
	index.write("<td>%s</td>" % fileset["name"])

	del fileset['name']

	for key, value in fileset.items():
		index.write("<td><img src='images/%s'></td>" % value)

	index.write("</tr>")
	return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
	fig, ax = plt.subplots()

	ax.plot(output.transpose(1, 0).detach().numpy(), '-')
	ax.plot(label.transpose(1, 0).detach().numpy(), '--')

	ax.set_xlim(0, 400)

	ax.grid(True)
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)

	writer.add_figure(namescope, fig, epoch)
