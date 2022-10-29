#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import os
import json
from PIL import Image
from traceback import print_exc

import torch
import torchvision.datasets as D
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

from modules.env import DATA_PATH

DATASETS = [
  'mnist', 
  'svhn', 
  'cifar10', 
  'cifar100', 
  'tiny-imagenet', 
  'imagenet-1k',
]


class TinyImageNet(Dataset):

  def __init__(self, root: str, split='train'):
    self.base_path = os.path.join(root, split)

    with open(os.path.join(root, 'synsets.txt'), encoding='utf-8') as fh:
      class_names = fh.read().strip().split('\n')
      assert len(class_names) == 1000
    class_name_to_label = {cname: i for i, cname in enumerate(class_names)}

    metadata = [ ]
    for class_name in os.listdir(self.base_path):
      dp = os.path.join(self.base_path, class_name, 'images' if split=='train' else '')
      for fn in os.listdir(dp):
        fp = os.path.join(dp, fn)
        metadata.append((fp, class_name_to_label[class_name]))

    self.metadata = metadata

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp)
    img = img.convert('RGB')
    im = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
    im = im / 255.0
    return im, tgt


class ImageNet_1k(Dataset):

  def __init__(self, root: str):
    self.base_path = os.path.join(root, 'val')

    fns = [fn for fn in os.listdir(self.base_path)]
    fps = [os.path.join(self.base_path, fn) for fn in fns]
    with open(os.path.join(root, 'image_name_to_class_id_and_name.json'), encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fn]['class_id'] for fn in fns]

    self.metadata = [x for x in zip(fps, tgts)]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp)
    img = img.convert('RGB')

    if 'use numpy':
      im = np.array(img, dtype=np.uint8).transpose(2, 0, 1)   # [C, H, W]
      im = im / np.float32(255.0)
    else:
      im = T.ToTensor()(img)

    return im, tgt


def normalize(X: torch.Tensor, dataset:str) -> torch.Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until put into model '''

  if 'imagenet' in dataset:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
  else:
    return X

  return TF.normalize(X, mean, std)       # [B, C, H, W]


def imshow(X, AX, title=''):
  DX = X - AX
  DX = (DX - DX.min()) / (DX.max() - DX.min())

  grid_X  = make_grid( X).permute([1, 2, 0]).detach().cpu().numpy()
  grid_AX = make_grid(AX).permute([1, 2, 0]).detach().cpu().numpy()
  grid_DX = make_grid(DX).permute([1, 2, 0]).detach().cpu().numpy()
  plt.subplot(131) ; plt.title('X')  ; plt.axis('off') ; plt.imshow(grid_X)
  plt.subplot(132) ; plt.title('AX') ; plt.axis('off') ; plt.imshow(grid_AX)
  plt.subplot(133) ; plt.title('DX') ; plt.axis('off') ; plt.imshow(grid_DX)
  plt.tight_layout()
  plt.suptitle(title)

  mng = plt.get_current_fig_manager()
  mng.window.showMaximized()    # 'QT4Agg' backend
  plt.show()


def get_dataloader(name, split='train', shuffle=False):
  transform_grey = T.Compose([
    T.Lambda(lambda x: x.convert('RGB')),
    T.ToTensor(),
  ])
  transform = T.ToTensor()
  datasets = {
    # 28 * 28
    'mnist'        : lambda: D.MNIST   (root=DATA_PATH, train=split=='train', transform=transform_grey, download=True),
    # 32 * 32
    'svhn'         : lambda: D.SVHN    (root=DATA_PATH, split=split         , transform=transform, download=True),
    'cifar10'      : lambda: D.CIFAR10 (root=DATA_PATH, train=split=='train', transform=transform, download=True),
    'cifar100'     : lambda: D.CIFAR100(root=DATA_PATH, train=split=='train', transform=transform, download=True),
    # 64 * 64
    'tiny-imagenet': lambda: TinyImageNet(root=os.path.join(DATA_PATH, 'tiny-imagenet-200'), split='train' if split=='train' else 'val'),
    # 224 * 224
    'imagenet-1k'  : lambda: ImageNet_1k(root=os.path.join(DATA_PATH, 'imagenet-1k')),
  }
  try:
    dataset = datasets[name]()
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=0)
  except Exception as e:
    print_exc()
