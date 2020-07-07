import os
from slbi_toolbox import SLBI_ToolBox
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_loader import load_data
import lenet
torch.backends.cudnn.benchmark = True
load_pth = torch.load('lenet.pth')
torch.cuda.empty_cache()
model = lenet.Net().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox(model.parameters(), lr=1e-1, kappa=1, mu=20, weight_decay=0)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
test_loader = load_data(dataset='MNIST', train=False, download=True, batch_size=64, shuffle=False)
#### test prune one layer
print('prune conv3')
print('acc before pruning')
evaluate_batch(model, test_loader, 'cuda')
print('acc after pruning')
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
evaluate_batch(model, test_loader, 'cuda')
print('acc after recovering')
optimizer.recover()
evaluate_batch(model, test_loader, 'cuda')
#### test prune two layers

print('prune conv3 and fc1')
print('acc before pruning')
evaluate_batch(model, test_loader, 'cuda')
print('acc after pruning')
optimizer.prune_layer_by_order_by_list(80, ['conv3.weight', 'fc1.weight'], True)
evaluate_batch(model, test_loader, 'cuda')
print('acc after recovering')
optimizer.recover()
evaluate_batch(model, test_loader, 'cuda')
