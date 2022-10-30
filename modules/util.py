#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/30

import torch


def kernel_norm(kernels: torch.Tensor):
  k_flat = kernels.flatten(start_dim=2)           # [C_out, C_in, H*W]
  k_min = k_flat.min(dim=-1)[0].unsqueeze_(-1).unsqueeze_(-1)
  k_max = k_flat.max(dim=-1)[0].unsqueeze_(-1).unsqueeze_(-1)
  return (kernels - k_min) / (k_max - k_min)
