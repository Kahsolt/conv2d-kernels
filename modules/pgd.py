#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import torch
import torch.nn.functional as F
from tqdm import tqdm

from scipy.stats import uniform

ATK_METH = [
  'pgd',
  'pgd_conv',
]


def pgd(model, images, labels, eps=0.03, alpha=0.001, steps=40, element_wise=True, **kwargs):
  normalizer = kwargs.get('normalizer', lambda _: _)

  images = images.clone().detach()
  labels = labels.clone().detach()

  adv_images = images.clone().detach()
  adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
  adv_images = torch.clamp(adv_images, min=0, max=1).detach()

  for _ in tqdm(range(steps)):
    adv_images.requires_grad = True
    outputs = model(normalizer(adv_images))

    if element_wise:
      loss = F.cross_entropy(outputs, labels, reduction='none')
      grad = torch.autograd.grad(loss, adv_images, grad_outputs=loss)[0]
    else:
      loss = F.cross_entropy(outputs, labels)
      grad = torch.autograd.grad(loss, adv_images)[0]

    v_loss = loss.mean().item()
    print('minimizing loss:', v_loss)
    if v_loss == 0.0: break

    with torch.no_grad():
      adv_images = adv_images.detach() - alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-eps, max=eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

  # assure valid rgb pixel
  adv_images = (adv_images * 255).round().div(255.0)

  return adv_images


def pgd_conv(layer, images, target:str, eps=0.03, alpha=0.001, steps=40, **kwargs):
  creterion = {
    # maximize the L-norm between original fm and new fm
    'L1': lambda x, y: (x - y).abs(),
    'L2': lambda x, y: (x - y).square(),

    # maximize the std between original fm and new fm in channel wise
    'L1_std'  : lambda x, y: (x - y).abs().std(),
    'L1_std_C': lambda x, y: (x - y).abs().std(dim=1),
    'L1_std_S': lambda x, y: (x - y).abs().std(dim=[2, 3]),
    # maximize the mean between original fm and new fm in channel wise
    'L1_mean'  : lambda x, y: (x - y).abs().mean(),
    'L1_mean_C': lambda x, y: (x - y).abs().mean(dim=1),
    'L1_mean_S': lambda x, y: (x - y).abs().mean(dim=[2, 3]),

    # maximize the dist between std of original fm and new fm in channel wise
    'std_L1'  : lambda x, y: (x.std() - y.std()).abs(),
    'std_C_L1': lambda x, y: (x.std(dim=1) - y.std(dim=1)).abs(),
    'std_S_L1': lambda x, y: (x.std(dim=[2, 3]) - y.std(dim=[2, 3])).abs(),
    # maximize the dist between mean of original fm and new fm in channel wise
    'mean_L1'  : lambda x, y: (x.mean() - y.mean()).abs(),
    'mean_C_L1': lambda x, y: (x.mean(dim=1) - y.mean(dim=1)).abs(),
    'mean_S_L1': lambda x, y: (x.mean(dim=[2, 3]) - y.mean(dim=[2, 3])).abs(),

    # maximize the std of new fm
    'std_fm'   : lambda x, y: x.std(),
    'std_C_fm' : lambda x, y: x.std(dim=1),
    'std_S_fm' : lambda x, y: x.std(dim=[2, 3]),
    'mean_fm'  : lambda x, y: x.mean(),
    'mean_C_fm': lambda x, y: x.mean(dim=1),
    'mean_S_fm': lambda x, y: x.mean(dim=[2, 3]),
  }.get(target)

  normalizer = kwargs.get('normalizer', lambda _: _)

  with torch.no_grad():
    fm = layer(normalizer(images))   # [B, C, H, W]

  images     = images.clone().detach()
  adv_images = images.clone().detach()
  adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
  adv_images = torch.clamp(adv_images, min=0, max=1).detach()

  last_v_loss = None
  for _ in tqdm(range(steps)):
    adv_images.requires_grad = True
    fm_hat = layer(normalizer(adv_images))   # [B, C, H, W]

    loss = creterion(fm_hat, fm)
    grad = torch.autograd.grad(loss, adv_images, grad_outputs=loss)[0]

    v_loss = loss.mean().item()
    print('maximizing loss:', v_loss)
    if v_loss == last_v_loss: break
    last_v_loss = v_loss
  
    with torch.no_grad():
      adv_images = adv_images.detach() + alpha * grad.sign()    # `+` for maximize
      delta = torch.clamp(adv_images - images, min=-eps, max=eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

  # assure valid rgb pixel
  adv_images = (adv_images * 255).round().div(255.0)

  return adv_images


def pgd_kernel_fixedpoint(kernel:torch.Tensor, X:torch.Tensor, alpha=0.001, steps=1000, ret_every=100):
  k_w, k_h = kernel.shape
  padding = (k_w//2, k_h//2)

  filters = kernel.unsqueeze(0).unsqueeze(0)     # [1, 1, k_w, k_h]

  for i in tqdm(range(steps)):
    X.requires_grad = True
    FX = F.conv2d(X, filters, padding=padding)   # [B=1, C=1, H, W]

    loss_diff = F.l1_loss(FX, X, reduction='none')
    loss_fx_overflow = FX * (FX>1) + -FX * (FX<0)       # FX should also range in [0, 1]
    loss = loss_diff + loss_fx_overflow
    grad = torch.autograd.grad(loss, X, grad_outputs=loss)[0]

    v_loss = loss.mean().item()
    print('minimizing loss:', v_loss)
    if v_loss == 0.0: raise StopIteration

    with torch.no_grad():
      if steps > 2000:
        X = X - alpha * grad.sign()
      else:
        X = X - alpha * grad.tanh()
      X = X.clamp(0.0, 1.0)

    if i % ret_every == 0:
      yield X, FX
