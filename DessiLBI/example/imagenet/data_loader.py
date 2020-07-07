#coding=utf-8
from __future__ import division, print_function
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
'''
This file is used to load data
Data used in this project includes MNIST, Cifar10 and ImageNet
'''


def load_data(dataset='ImageNet', train=False, download=True, transform=None, batch_size=1, shuffle=True):
	if dataset == 'ImageNet':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
		if train:
			dataset = torchvision.datasets.ImageFolder('/home/lc/data/ILSVRC2012/train', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
		else:
			dataset = torchvision.datasets.ImageFolder('/home/lc/data/ILSVRC2012/val/', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=4)
	else:
		print('No such dataset')
	return data_loader,  dataset.class_to_idx

if __name__ == '__main__':
	data_loader = load_data()
	for data, target in data_loader:
		print(data)
		print(target) 
		stop

