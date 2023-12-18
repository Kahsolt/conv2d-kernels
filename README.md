# conv2d-kernels

    Interactive experiments on pretrained Conv2d layer weights.

----

Conv2d layers are learnable 2D signal filters, it self is indeed a function-let, so what can it actually do? 🤔

### Apps

#### filter

⚪ inspect into featur maps: what does the **first** conv2d layer do in the well-known classifiers?

![img/filter.png](img/filter.png)

#### kernel

⚪ inspect into conv2d kernels: what are the learned geometrical basis?

The first Conv2d layer kernels of a pretrained model:

![img/kernel.png](img/kernel.png)

#### cluster

⚪ grouping kernels: many kernels seems to be similar thus redundant?

The first Conv2d layer kernels of a pretrained model (and centroids):

| kernels | kernel centroids |
| :-: | :-: |
| ![img/resnet18-conv1-kernels.png](img/resnet18-conv1-kernels.png) | ![img/resnet18-conv1-kernel-centroids.png](img/resnet18-conv1-kernel-centroids.png) |

#### attack

⚪ PGD adversarial attack: what if we attack a single conv2d layer?

![img/attack.png](img/attack.png)

#### fixedpoint

⚪ mathematical property of the well-known image kernels: what are the fixed points of a 2d kernel?

![img/fixedpoint.png](img/fixedpoint.png)


#### resources download

- The reprocessed ImageNet-1k dataset can be downloaded here: [https://pan.quark.cn/s/373b488d101e](https://pan.quark.cn/s/373b488d101e)
  - NOTE: It is a subset of 1k images from validation split of original intact ImageNet dataset
- Tiny-ImageNet can be found here: [tiny-imagenet-200](https://tiny-imagenet.herokuapp.com)

----

by Armit
2022/10/28 
