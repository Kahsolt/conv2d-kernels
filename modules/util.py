#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/30

import torch

def minmax_norm_channel_wise(x: torch.Tensor):
  ''' x: [B, C, H, W] '''
  
  x_flat = x.flatten(start_dim=2)           # [B, C, H*W]
  x_min = x_flat.min(dim=-1)[0].unsqueeze_(-1).unsqueeze_(-1)
  x_max = x_flat.max(dim=-1)[0].unsqueeze_(-1).unsqueeze_(-1)
  return (x - x_min) / (x_max - x_min)
