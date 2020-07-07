from __future__ import division, print_function
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
'''
This file is used to load data
Data used in this project includes MNIST
'''


def load_data(dataset='MNIST', train=True, download=True, transform=None, batch_size=1, shuffle=True):
	if dataset == 'MNIST':
		data_loader = torch.utils.data.DataLoader(datasets.MNIST('data/MNIST', train=train, download=download, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=batch_size, shuffle=shuffle)
	else:
		print('No such dataset')
	return data_loader


