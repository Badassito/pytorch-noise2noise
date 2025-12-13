import argparse

import torch.backends.cudnn as cudnn
from train import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the unet network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='resnet', dest='scope')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')

parser.add_argument('--name_data', type=str, default='bsd500', dest='name_data')
parser.add_argument('--dir_data', default='./datasets', dest='dir_data')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=100, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=1e-4, dest='lr_G')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', type=float, default=0.5, dest='beta1')

parser.add_argument('--ny_load', type=int, default=256, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=256, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=1, dest='nch_load')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')

parser.add_argument('--num_freq_disp', type=int,  default=1, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')

# Performance optimization arguments
parser.add_argument('--use_amp', action='store_true', default=True, dest='use_amp', help='Use automatic mixed precision training')
parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='Disable automatic mixed precision training')
parser.add_argument('--use_compile', action='store_true', default=False, dest='use_compile', help='Use torch.compile() for model compilation')
parser.add_argument('--use_grad_checkpoint', action='store_true', default=True, dest='use_grad_checkpoint', help='Use gradient checkpointing to save memory')
parser.add_argument('--no_grad_checkpoint', action='store_false', dest='use_grad_checkpoint', help='Disable gradient checkpointing')
parser.add_argument('--use_scheduler', action='store_true', default=True, dest='use_scheduler', help='Use cosine annealing learning rate scheduler')
parser.add_argument('--no_scheduler', action='store_false', dest='use_scheduler', help='Disable learning rate scheduler')
parser.add_argument('--save_images', action='store_true', default=False, dest='save_images', help='Save images during training (slower)')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    main()
