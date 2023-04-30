from coldbin import Unet, GaussianDiffusion, Trainer
import os
import errno
import shutil
import argparse
import torch 

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass



parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=200, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='results_2013/', type=str)
parser.add_argument('--data_path', default='/path/to/dibco/', type=str, required=True)
parser.add_argument('--dataset', default='2013', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l2', type=str)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torch
torch.manual_seed(0)

args = parser.parse_args()
print(args)


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = args.image_size,
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))
trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = args.image_size,
    batch_size = args.batch_size,
    train_lr = 2e-5,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset= args.dataset,
    dataset_split = 'test',
)

avg_psnr_full, avg_psnr_direct = trainer.sample_and_save_val()
print(f'Avg PSNR Full: {avg_psnr_full}')
print(f'Avg PSNR Direct: {avg_psnr_direct}')