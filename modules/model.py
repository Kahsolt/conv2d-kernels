#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/28 

import torch.nn as nn
import torchvision.models as M

from modules.env import device

MODELS = [
  'resnet18',
  'resnet34',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnext50_32x4d',
  'resnext101_32x8d',
  'resnext101_64x4d',
  'wide_resnet50_2',
  'wide_resnet101_2',

  'densenet121',
  'densenet161',
  'densenet169',
  'densenet201',

  'mobilenet_v3_large',
  'mobilenet_v3_small',
]


def get_first_conv2d_layer(model):
  if isinstance(model, M.ResNet):
    return model.conv1.to(device)
  if isinstance(model, M.DenseNet):
    return model.features['conv0'].to(device)
  
  for layer in model.modules():
    if isinstance(layer, nn.Conv2d):
      return layer.to(device)


def get_model(name):
  model = getattr(M, name)(pretrained=True)
  
  return model.to(device)
