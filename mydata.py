from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import pandas as pd
import torch as th
import pickle
import random

from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, normalize_complex, ifft2_m, fft2_m, normalize
from torchvision.transforms.functional import center_crop
from torchvision.transforms import v2
from PIL import Image

def select_slices(filename, number_of_slices):
    if number_of_slices % 2 == 0:
        return [(filename, slice) for slice in range(-(number_of_slices // 2), number_of_slices // 2)]
    return [(filename, slice) for slice in range(-(number_of_slices // 2), number_of_slices // 2 + 1)]

def select_brain_slices(filename, number_of_slices):
    return [(filename, slice) for slice in range(number_of_slices)]

class MRIDataset(DataLoader):
    def __init__(self, root, split='train', is_complex=False):
        self.root = root
        self.is_complex = is_complex

        self.filenames = open(f'./split/corpd/{split}.txt', 'r').read().splitlines()
        self.data = []
        for filename in self.filenames:
            self.data.extend(select_slices(filename, 15))

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = file['reconstruction_rss'][len(file['reconstruction_rss']) // 2 + slice]
        data = data - np.min(data)
        return data / np.max(data)

    def __len__(self):
        return len(self.data)
    
class MRITestDataset(DataLoader):
    def __init__(self, root, split='test', fat_suppression=False, is_complex=True):
        self.root = root
        self.is_complex = is_complex
        if split == 'test' and fat_suppression == False:
            self.filenames = open(f'./split/corpd/validation.txt', 'r').read().splitlines()
            self.filenames += open(f'./split/corpd/test.txt', 'r').read().splitlines()
        elif split == 'test' and fat_suppression == True:
            self.filenames = open(f'./split/corpdfs/test.txt', 'r').read().splitlines()
        random.seed(42)
        self.filenames = random.sample(self.filenames, 50)
        open(f'./split/{"corpdfs" if fat_suppression else "corpd"}/test_selected.txt', 'w').write('\n'.join(self.filenames))
        self.data = []
        for filename in self.filenames:
            self.data.extend(select_slices(filename, 10))

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = normalize(th.from_numpy(file['reconstruction_rss'][len(file['reconstruction_rss']) // 2 + slice]))
        return data

    def __len__(self):
        return len(self.data)
    
class MRITestBrainDataset(DataLoader):
    def __init__(self, root, split='test', is_complex=True):
        self.root = root
        self.is_complex = is_complex
        self.filenames = open(f'./split/brain/test_selected.txt', 'r').read().splitlines()
        self.data = []
        for filename in self.filenames:
            self.data.extend(select_brain_slices(filename, 5))


    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = normalize(center_crop(th.from_numpy(file['reconstruction_rss'][slice]), (320, 320)))
        return data

    def __len__(self):
        return len(self.data)
    

class MRITestProstateDataset(DataLoader):
    def __init__(self, root, split='test', is_complex=True):
        self.root = root
        self.is_complex = is_complex
        self.filenames = os.listdir(root)
        #random.seed(42)
        #self.filenames = random.sample(self.filenames, 100)
        #open(f'./split/brain/test_selected.txt', 'w').write('\n'.join(self.filenames))
        self.data = []
        for filename in self.filenames[:-1]:
            self.data.extend(select_slices(filename, 11))
        self.data.extend(select_slices(self.filenames[-1], 5))


    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = normalize(th.from_numpy(file['reconstruction_rss'][slice]))
        return data

    def __len__(self):
        return len(self.data)


class MRIDataset_infer(DataLoader):
    def __init__(self, root, sort=True, split='train', is_complex=False):
        self.root = root
        self.is_complex = is_complex

        self.filenames = open(f'./split/corpd/{split}.txt', 'r').read().splitlines()
        self.data = [select_slices(filename, 15) for filename in self.filenames]

    def __getitem__(self, idx):
        filename = self.data[idx][0]
        slice = self.data[idx][1]
        file = h5py.File(self.root / filename, 'r')
        data = file['reconstruction_rss'][len(file['reconstruction_rss']) // 2 + slice]
        data = data - np.min(data)
        return data / np.max(data), filename

    def __len__(self):
        return len(self.data)


def create_dataloader(configs, evaluation=False, sort=True, fat_suppression=False):
    shuffle = True if not evaluation else False
    train_dataset = MRIDataset(Path('/srv/local/lg/FAST_MRI') / f'singlecoil_train')
    val_dataset = MRITestDataset(Path('/srv/local/lg/FAST_MRI') / f'singlecoil_val', fat_suppression=fat_suppression)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs.training.batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=configs.training.batch_size,
        shuffle=False,
        # shuffle=True,
        drop_last=True
    )
    return train_loader, val_loader

def create_brain_dataloader():
    val_dataset = MRITestBrainDataset(Path('/srv/local/lg/Brain_MRI') / f'multicoil_test_full')
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return val_loader

def create_prostate_dataloader():
    val_dataset = MRITestProstateDataset(Path('/data/glaszner/prostate') / f'test_T2_1')
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )
    return val_loader


class CelebADataset(DataLoader):
    def __init__(self, root, size):
        self.root = root
        self.data = [str(i) for i in range(30_000)]

        if os.path.exists(root + '/resized'):
            return

        os.mkdir(root + '/resized')
        transform = v2.Compose([
            v2.PILToTensor(),
            v2.Grayscale(),
            v2.Resize((320, 320), antialias=True)
            ])
        for filename in self.data:
            img = Image.open(self.root + '/' + filename + '.jpg')
            img = transform(img)
            img = img - th.min(img)
            img = img / th.max(img)
            np.save(root + '/resized/' + filename, img.numpy())


    def __getitem__(self, idx):
        filename = self.data[idx]
        data = np.load(self.root + '/resized/' + filename + '.npy')
        return data 

    def __len__(self):
        return len(self.data)

def create_celeba_dataloader(size):
    train_dataset = CelebADataset('/srv/local/lg/CelebA-HQ-img', size)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    return train_loader

def create_dataloader_regression(configs, evaluation=False):
  shuffle = True if not evaluation else False
  train_dataset = fastmri_knee(Path(configs.root) / f'knee_{configs.image_size}_train')
  val_dataset = fastmri_knee_infer(Path(configs.root) / f'knee_{configs.image_size}_val')

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=configs.batch_size,
    shuffle=shuffle,
    drop_last=True
  )
  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=configs.batch_size,
    shuffle=False,
    drop_last=True
  )
  return train_loader, val_loader