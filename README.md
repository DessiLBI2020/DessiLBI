# Exploring Structural Sparsity of Deep Networks via Inverse Scale Spaces. TPAMI 2022
[Arxiv Link](https://arxiv.org/pdf/1905.09449.pdf)

TPAMI 2022 [Early Access](https://ieeexplore.ieee.org/document/9762064)

Yanwei Fu, Chen Liu, Donghao Li, Zuyuan Zhong, Xinwei Sun, Jinshan Zeng, Yuan Yao

 (Extended Version of DessiLBI: Exploring Structural Sparsity of Deep Networks via Differential Inclusion Paths ICML2020)

## Abstract
The great success of deep neural networks is built upon their over-parameterization, which smooths the optimization
landscape without degrading the generalization ability. Despite the benefits of over-parameterization, a huge amount of parameters
makes deep networks cumbersome in daily life applications. On the other hand, training neural networks without over-parameterization
faces many practical problems, e.g., being trapped in local optimal. Though techniques such as pruning and distillation are developed,
they are expensive in fully training a dense network as backward selection methods, and there is still a void on systematically exploring
forward selection methods for learning structural sparsity in deep networks. To fill in this gap, this paper proposes a new approach based
on differential inclusions of inverse scale spaces. Specifically, our method can generate a family of models from simple to complex ones
along the dynamics via coupling a pair of parameters, such that over-parameterized deep models and their structural sparsity can be
explored simultaneously. This kind of differential inclusion scheme has a simple discretization, dubbed Deep structure splitting Linearized
Bregman Iteration (DessiLBI), whose global convergence in learning deep networks could be established under the Kurdyka-Łojasiewicz
framework. Particularly, we explore several applications of DessiLBI, including finding sparse structures of networks directly via the
coupled structure parameter and growing networks from simple to complex ones progressively. Experimental evidence shows that our
method achieves comparable and even better performance than the competitive optimizers in exploring the sparse structure of several
widely used backbones on the benchmark datasets. Remarkably, with early stopping, our method unveils “winning tickets” in early epochs:
the effective sparse network structures with comparable test accuracy to fully trained over-parameterized models, that are further
transferable to similar alternative tasks. Furthermore, our method is able to grow networks efficiently with adaptive filter configurations,
demonstrating a good performance with much less computational cost.
## Short Version
DessiLBI: Exploring Structural Sparsity of Deep Networks via Differential Inclusion Paths (ICML 2020) [Link](http://proceedings.mlr.press/v119/fu20d/fu20d.pdf)



# DessiLBI

This is the project page for DessiLBI. It is an optimization toolbox for <a href="https://pytorch.org/"> <b>Pytorch.</b>

# Prerequisites
Pytorch 1.0 +
Numpy  
Python 3+
CUDA
# Installation
It is install-free, just put the slbi_opt.py and slbi_toolbox.py into the the project folder and import them. 
# Examples
We give examples for train network and prune network on MNIST dataset. These codes are in the example folder. The following codes are some usage of this toolbox.

# Quick start: running an example of training a  network

To train the LeNet network, please cd to the corresponding folder and just simply run   
```bash
 python ./example/train/train_lenet.py  
```

# Train your own network:
To initialize the toolbox, the following codes are needed.
```python
from slbi_toolbox import SLBI_ToolBox
import torch
optimizer = SLBI_ToolBox(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
```
For training a neural network, the process is similar to one that uses built-in optimizer
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

For pruning a neural network, the code is as follows.   

```python
optimizer.update_prune_order(epoch)
optimizer.prune_layer_by_order_by_list(percent, layer_name)
```

# Pruning Network Example
This example is about Lenet using Split LBI, the code is in example/prune. We have put one pretrained model and optimizer dict in this folder. It shows the pruned results using Split LBI.  
To run this demo, please download and cd to this folder and run the following code:  
```bash
 python prune_lenet.py 
 ```
# ImageNet Training Example
This part of code is included in example/imagenet. To do this demo, run 
```bash
 python train_imagenet_slbi.py
```
# Lottery Training Example
The part of lottery training code is included in example/lottery. To conduct this demo, please refer to the readme file under  example/lottery.

 # Weight Pruning Example of Extended Version
This part of code is included in example/cifar_prune. To conduct this demo, please refer to the readme file under  example/cifar_prune.
 To run training and pruning, you can run
 ```shell
 sh run_lbi.sh
 ```
 To visualize the sparse structue, please use vis.sh
 
 # Filter Visualization Example
[Pytorch CNN Visualization](https://github.com/utkuozbulak/pytorch-cnn-visualizations) is used to visualize the filters. We give one example for visualization of LeNt on MNIST in example/mnist_plot_example
 

