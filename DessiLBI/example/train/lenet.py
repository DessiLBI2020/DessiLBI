#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
		self.fc1 = nn.Linear(120, 84)
		self.fc2 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.relu(self.conv3(x))
		x = x.view(-1, 120)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
