#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/28 

from argparse import ArgumentParser
from collections import Counter

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.model import MODELS, get_model, get_first_conv2d_layer
from modules.plot import imshow
from modules.util import minmax_norm_channel_wise


def optimal_number_of_clusters(wcss, n_clust_min, n_clust_max, n_clust_step):
  x1, y1 = n_clust_min, wcss[0]
  x2, y2 = n_clust_max, wcss[len(wcss)-1]

  distances = []
  for i in range(len(wcss)):
    x0 = i+n_clust_min
    y0 = wcss[i]
    numerator = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distances.append(numerator/denominator)

  return distances.index(max(distances)) * n_clust_step + n_clust_min


def cluster(args):
  model = get_model(args.model).eval()
  layer = get_first_conv2d_layer(model)
  kernels = layer.weight.detach()         # [C_out, C_in, H, W]
  C_out, C_in, H, W = kernels.shape

  if 'draw kernels':
    for i, k in enumerate(kernels):
      print(f'kernel-{i}: mean={k.mean()}, std={k.std()}')
    kernels_n = minmax_norm_channel_wise(kernels) if args.show_norm else kernels
    imshow(kernels_n, 'kernels', maximize=True)

  data = kernels.flatten(start_dim=1)     # [C_out, C_in*H*W]

  # try & select model by `wcss` metric
  wcss = []
  for n_clust in range(args.n_clust_min, args.n_clust_max, args.n_clust_step):
    model = KMeans(n_clusters=n_clust)
    model.fit(data)
    print(f'n_clust = {n_clust}, inertia = {model.inertia_}')
    wcss.append(model.inertia_)
    del model
  
  if 'show wcss':
    plt.plot(wcss)
    plt.title('wcss')
    plt.show()

  # choose & save the best model
  n_clust = optimal_number_of_clusters(wcss, args.n_clust_min, args.n_clust_max, args.n_clust_step)
  print(f'auto decision: n_clust={n_clust}')
  model = KMeans(n_clusters=n_clust)
  model.fit(data)
  print(f'inertia = {model.inertia_}')

  # infer
  n_clust = model.n_clusters
  print(f'inertia = {model.inertia_}')
  dist = model.transform(data)
  pred = dist.argmin(axis=-1)  
  pred = pred.tolist()

  if 'pred':
    plt.scatter(range(len(pred)), pred)
    plt.title('pred')
    plt.show()

  if 'draw histogram & sorted freqs':
    plt.subplot(211) ; plt.title('hist') ; plt.hist(pred, bins=n_clust)
    plt.subplot(212) ; plt.title('freq') ; plt.plot(sorted(Counter(pred).values(), reverse=True))
    plt.show()

  centroids = model.cluster_centers_
  if 'draw centroid kernels':
    centroid_kernels = centroids.reshape(n_clust, C_in, H, W)
    for i, k in enumerate(centroid_kernels):
      print(f'kernel-{i}: mean={k.mean()}, std={k.std()}')
    c_ker = torch.from_numpy(centroid_kernels)
    kernels_n = minmax_norm_channel_wise(c_ker) if args.show_norm else c_ker
    imshow(kernels_n, 'centroid kernels', maximize=True)
  
  if 'draw distmap':
    distmat = np.abs(centroids[None, :, :] - centroids[:, None, :]).mean(axis=-1)
    sns.heatmap(distmat)
    plt.title('dist map')
    plt.show()

  if 'draw dim reduce':
    pca  = PCA(n_components=2).fit_transform(data)
    kpca = KernelPCA(n_components=2, kernel='rbf').fit_transform(data)
    tsne = TSNE(n_components=2, verbose=True, learning_rate='auto').fit_transform(data)
    lle  = LocallyLinearEmbedding(n_components=2).fit_transform(data)
    plt.subplot(221) ; plt.title('PCA')  ; plt.scatter(pca [:, 0], pca [:, 1], c=pred)
    plt.subplot(222) ; plt.title('KPCA') ; plt.scatter(kpca[:, 0], kpca[:, 1], c=pred)
    plt.subplot(223) ; plt.title('TSNE') ; plt.scatter(tsne[:, 0], tsne[:, 1], c=pred)
    plt.subplot(224) ; plt.title('LLE')  ; plt.scatter(lle [:, 0], lle [:, 1], c=pred)
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model to inspect')
  parser.add_argument('--n_clust_min',  type=int, default=2)
  parser.add_argument('--n_clust_max',  type=int, default=32)
  parser.add_argument('--n_clust_step', type=int, default=1)
  parser.add_argument('--show_norm',    action='store_true', help='min-max normalize kernels before show')
  args = parser.parse_args()
  
  cluster(args)
