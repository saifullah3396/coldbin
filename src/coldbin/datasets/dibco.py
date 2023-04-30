from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF
import random
from pathlib import Path
import pickle
import imageio
import os 
import matplotlib.pyplot as plt
import re
import numpy as np 
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
class DibcoDataset(Dataset):
    def __init__(self, data_path, dataset="2013", image_size=256, split='train', save_images_in_pickle=False):
        self.data_path = data_path
        self.split = split
        data_path = Path(self.data_path) / dataset / f"{split}.pickle"
        self.image_size = image_size
        self.save_images_in_pickle = save_images_in_pickle

        if not data_path.exists():
            # print("Reading data into pickle file...")
            # print('data_path.parent', data_path.parent)
            filenames = sorted_alphanumeric(os.listdir(data_path.parent / split))
            self.data = []
            for filename in filenames:
                if self.save_images_in_pickle:
                    image = imageio.imread(data_path.parent / split / filename)
                    gt_image = imageio.imread(data_path.parent / f'{split}_gt' / filename)
                else:
                    image = data_path.parent / split / filename
                    gt_image = data_path.parent / f'{split}_gt' / filename
                self.data.append(dict(image=image, gt_image=gt_image, filename=filename))
            with open(data_path, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            print("Reading data from pickle file...")
            self.data = None
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)

        if self.split == 'train':
            self.tfs = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(0.2, 0.2, 0.1, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.gt_tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.unnormalize = transforms.Compose([
            UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def transform_images(self, image, gt_image):
        if self.split == 'train':
            # random coloring
            if random.random() > 0.95:
                color = list(np.random.choice(range(256), size=3))
                image[:, :, 0][gt_image[:, :, 0]==0] = color[0]
                image[:, :, 1][gt_image[:, :, 1]==0] = color[1]
                image[:, :, 2][gt_image[:, :, 2]==0] = color[2]

        image = to_pil_image(image)
        gt_image = to_pil_image(gt_image)

        # apply data augmentation
        if self.split == 'train':
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.image_size, self.image_size))
            image = TF.crop(image, i, j, h, w)
            gt_image = TF.crop(gt_image, i, j, h, w)

            # random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt_image = TF.hflip(gt_image)

            # random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                gt_image = TF.vflip(gt_image)

        image = self.tfs(image)
        gt_image = self.gt_tfs(gt_image)

        return image, gt_image

    def __getitem__(self, idx):
        sample = self.data[idx]
        if not self.save_images_in_pickle:
            sample['image'] = imageio.imread(sample['image'])
            sample['gt_image'] = imageio.imread(sample['gt_image'])
        sample['image'], sample['gt_image'] = self.transform_images(
            sample['image'], sample['gt_image'])
        return sample

    def __len__(self):
        return len(self.data)
