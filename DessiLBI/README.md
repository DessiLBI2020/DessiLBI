# DessiLBI

This is the project page for DessiLBI. It is an optimization toolbox for <a href="https://pytorch.org/"> <b>Pytorch.</b>

# Prerequisites
Pytorch 1.0 +
Numpy  
Python 3  
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


