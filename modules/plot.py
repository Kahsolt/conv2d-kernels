#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/29 

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def imshow(X: torch.Tensor, title='', maximize=False):
  nrow = int(X.shape[0] ** 0.5)
  grid_X = make_grid(X, nrow=nrow).permute([1, 2, 0]).detach().cpu().numpy()
  plt.imshow(grid_X)
  plt.axis('off')
  plt.tight_layout()
  plt.suptitle(title)

  if maximize:
    try:
      mng = plt.get_current_fig_manager()
      mng.window.showMaximized()    # 'QT4Agg' backend
    except: pass
  plt.show()


def imshow_adv(X: torch.Tensor, AX: torch.Tensor, title='', maximize=False):
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

  if maximize:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()    # 'QT4Agg' backend
  
  plt.show()

