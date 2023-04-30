import copy
import torch
from functools import partial
from torch.cuda import amp
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import utils
import math 
import numpy as np 

import os
from .datasets.dibco import DibcoDataset

# trainer class
import os
import errno
def create_folder(path):
    from pathlib import Path
    if not Path(path).exists():
        Path(path).mkdir(parents=True)

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('.module', '')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def cycle(dl):
    while True:
        for data in dl:
            yield data

def psnr(img1, img2):
    """
    Count PSNR of two images
    Args:
        img1 (np.array): first image
        img2 (np.array): second image
    Returns:
        p (int): the PSNR value 
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    p = (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    
    return p

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        data_path,
        *,
        ema_decay = 0.995,
        image_size = 128,
        batch_size = 128,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 1,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        load_path = None,
        dataset = None,
        dataset_split = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = DibcoDataset(data_path=Path(data_path), dataset=dataset, image_size=image_size, split=dataset_split)
        if dataset_split == 'train':
            self.dl = cycle(data.DataLoader(self.ds, batch_size = batch_size, shuffle=shuffle, pin_memory=True, num_workers=16, drop_last=True))

            self.val_ds = DibcoDataset(data_path=Path(data_path), dataset=dataset, image_size=image_size, split='test')
            self.val_dl = data.DataLoader(self.val_ds, batch_size = 16, shuffle=False, pin_memory=True, num_workers=16, drop_last=False)
        else:
            self.val_dl = data.DataLoader(self.ds, batch_size = 16, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)


    def train(self):
        import tqdm
        scaler = None 
        if self.fp16:
            try:
                from torch.cuda.amp import autocast
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

            if scaler is None:
                from torch.cuda.amp.grad_scaler import GradScaler

                scaler = GradScaler(enabled=True)

        acc_loss = 0
        pbar = tqdm.tqdm(total=self.train_num_steps)
        with pbar:
            while self.step < self.train_num_steps:
                self.opt.zero_grad()

                u_loss = 0
                for i in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    if self.fp16:
                        with autocast(enabled=True):
                            loss = torch.mean(self.model(data['gt_image'].cuda(), data['image'].cuda()))
                            # if self.step % 100 == 0:
                            #     print(f'{self.step}: {loss.item()}')
                            u_loss += loss.item()
                            loss = loss / self.gradient_accumulate_every
                        if scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    else:
                        loss = torch.mean(self.model(data['gt_image'].cuda(), data['image'].cuda()))
                        # if self.step % 100 == 0:
                        #     print(f'{self.step}: {loss.item()}')
                        u_loss += loss.item()
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()

                if self.fp16:
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    self.opt.step()

                acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    acc_loss = acc_loss/(self.save_and_sample_every+1)
                    print(f'Mean of last {self.step}: {acc_loss}')
                    acc_loss=0

                    self.save()
                    if self.step % (self.save_and_sample_every * 1) == 0:
                        avg_psnr_direct = self.sample_and_save_val_direct_only(step=self.step)
                        print(f'Avg PSNR Direct {self.step}: {avg_psnr_direct}')

                        output = f'{self.results_folder}/psnr_direct.txt'
                        with open(output, 'a') as f:
                            f.write(f'{self.step} {avg_psnr_direct}\n')

                    if self.step % (self.save_and_sample_every * 10) == 0:
                        print("Running validation step: ", self.step)
                        self.save(self.step)

                self.step += 1
                pbar.update(1)

        print('training completed')
        
    def sample_and_save_val(self, noise=0):
        import tqdm
        gt_folder = f'{self.results_folder}/gt'
        create_folder(gt_folder)

        out_folder = f'{self.results_folder}/out'
        create_folder(out_folder)

        direct_recons_folder = f'{self.results_folder}/dir_recons'
        create_folder(direct_recons_folder)

        output_dict = {}
        output_dict_dir_recons = {}
        output_dict_gt = {}

        for batch in tqdm.tqdm(self.val_dl):
            og_img = batch['image'].cuda()
            bs = og_img.shape[0]
            xt, direct_recons, all_images = self.ema_model.module.gen_sample(batch_size=bs, img=og_img,
                                                                                noise_level=noise)
            for i in range(direct_recons.shape[0]):
                f = batch['filename'][i]
                image_idx = f.split('_')[0]
                py = int(f.split('_')[1])
                px = int(f.split('_')[2])
                h = int(f.split('_')[3])
                w = int(f.split('_')[4].replace('.png', ''))

                direct_recons[i] = self.ds.unnormalize(direct_recons[i])
                all_images[i] = self.ds.unnormalize(all_images[i])

                if image_idx not in output_dict_dir_recons:
                    output_dict[image_idx] = []
                    output_dict_dir_recons[image_idx] = []
                    output_dict_gt[image_idx] = []
                output_dict[image_idx].append([py, px, all_images[i].detach().cpu(), h, w])
                output_dict_dir_recons[image_idx].append([py, px, direct_recons[i].detach().cpu(), h, w])
                output_dict_gt[image_idx].append([py, px, batch['gt_image'][i].detach().cpu(), h, w])

        def reconstruct(output_dict):
            import torchvision.transforms.functional as TF
            reconstructed = {}
            for image_idx, values in output_dict.items():
                y_max = max([v[0] for v in values])
                x_max = max([v[1] for v in values])
                image = torch.zeros((3, y_max+self.image_size,x_max+self.image_size))
                import torchvision.transforms.functional as TF
                for v in values:
                    image[:, v[0]:v[0]+self.image_size, v[1]:v[1]+self.image_size] = v[2]
                
                # crop image to original height and width
                image = image[:3, :v[3], :v[4]]
                image = image.detach().cpu()
                reconstructed[image_idx] = image

            return reconstructed            

        def save(output_dict, folder):
            for image_idx, image in output_dict.items():
                import imageio
                utils.save_image(image, f'{folder}/{image_idx}.png')

        output_dict = reconstruct (output_dict)
        output_dict_dir_recons = reconstruct (output_dict_dir_recons)
        output_dict_gt = reconstruct (output_dict_gt)
        total_psnr_full = 0       
        total_psnr_direct = 0
        qo = 0
        # count psnrs
        for image_idx in output_dict_dir_recons.keys():
            import matplotlib.pyplot as plt 
            import cv2
            direct_recons = cv2.cvtColor(output_dict_dir_recons[image_idx].permute(1, 2, 0).numpy().clip(0, 1.0), cv2.COLOR_RGB2GRAY)
            full_recons = cv2.cvtColor(output_dict[image_idx].permute(1, 2, 0).numpy().clip(0, 1.0), cv2.COLOR_RGB2GRAY)
            gt = output_dict_gt[image_idx].permute(1, 2, 0).numpy()
            gt = cv2.cvtColor(output_dict_gt[image_idx].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
            full_recons = (full_recons > 0.5) * 1.0
            direct_recons = (direct_recons > 0.5) * 1.0
            gt = (gt > 0.5) * 1.0
            total_psnr_full+=psnr(full_recons, gt)
            total_psnr_direct+=psnr(direct_recons, gt)            
            print('psnr(full_recons, gt)', psnr(full_recons, gt))
            print('psnr(direct_recons, gt)', psnr(direct_recons, gt))
            qo+=1
            import imageio
            imageio.imwrite(f'{out_folder}/{image_idx}.png', full_recons)
            imageio.imwrite(f'{direct_recons_folder}/{image_idx}.png', direct_recons)
            imageio.imwrite(f'{gt_folder}/{image_idx}.png', gt)
        avg_psnr_full = total_psnr_full / qo
        avg_psnr_direct = total_psnr_direct / qo
        return avg_psnr_full, avg_psnr_direct