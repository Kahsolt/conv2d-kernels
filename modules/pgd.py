#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import torch
import torch.nn.functional as F
from tqdm import tqdm


def pgd(model, images, labels, eps=0.03, alpha=0.001, steps=40, element_wise=True, **kwargs):
  normalizer = kwargs.get('', lambda _: _)

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

    with torch.no_grad():
      adv_images = adv_images.detach() - alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-eps, max=eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

  # assure valid rgb pixel
  adv_images = (adv_images * 255).round().div(255.0)

  return adv_images
