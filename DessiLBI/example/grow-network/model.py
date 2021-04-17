#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import time
import datetime
import os
import logging
import matplotlib.pyplot as plt
import argparse


class Conv_3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(Conv_3, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,
                               stride=stride, padding=padding)

    def forward(self, x):
        out = F.relu(self.conv(x))
        return out



class Conv_bn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(Conv_bn, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        #out = F.relu(self.conv(x))
        return out


#[1,2,3,3,3], [[10], [10,10], [10,10,10], [10,10,10],[10,10,10]]
class VGG(nn.Module):
    def __init__(self, block, num_blocks, num_filters, num_classes=1000, resolution=224, init_normal=True):
        super(VGG, self).__init__()
        self.num_class = num_classes
        self.in_planes = 3
        self.out_resolution = int(resolution/2**4)
        self.is_vgg19 = (len(num_filters[2]) == 4)
        self.layer1 = self._make_layer(block, num_filters[0], num_blocks[0])
        self.layer2 = self._make_layer(block, num_filters[1], num_blocks[1])
        self.layer3 = self._make_layer(block, num_filters[2], num_blocks[2])
        self.layer4 = self._make_layer(block, num_filters[3], num_blocks[3])
        self.layer5 = self._make_layer(block, num_filters[4], num_blocks[4])
        self.linear1 = nn.Linear(self.out_resolution * self.out_resolution * num_filters[-2][-1], num_filters[-1][0])
        self.relu = nn.ReLU(inplace=True)
        if self.is_vgg19:
            self.linear2 = nn.Linear( num_filters[-1][0], self.num_class)
#             self.linear2 = nn.Linear( num_filters[-1][0], num_filters[-1][1])
#             self.linear3 = nn.Linear( num_filters[-1][1], self.num_class)
        else:
            self.linear2 = nn.Linear( num_filters[-1][0], self.num_class)
            self.bn = nn.BatchNorm1d(num_filters[-1][-1])
            self.drop_out1 = nn.Dropout(p=0.5)
            self.drop_out2 = nn.Dropout(p=0.5)

        if init_normal:
            self._initialize_weights()


    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes[i]))
            self.in_planes = planes[i] * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = F.max_pool2d(out, 2)
        
        out = self.layer2(out)
        out = F.max_pool2d(out, 2)
        
        out = self.layer3(out)
        out = F.max_pool2d(out, 2)
        
        out = self.layer4(out)
        out = F.max_pool2d(out, 2)
        
        out = self.layer5(out)
        #out = F.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)
        if self.is_vgg19:
            out = self.relu(self.linear1(out))
            out = self.linear2(out)
#             out = self.relu(self.linear2(out))
#             out = self.linear3(out)
        else:
            out = self.drop_out1(out)
            out = self.linear1(out)
            out = self.relu(self.bn(out))
            out = self.drop_out2(out)
            out = self.linear2(out)
        return out    




'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_Light(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_Light, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                diff = planes - in_planes
                if diff%2 == 0:
                    d1 = int(diff/2)
                    d2 = int(diff/2)
                else:
                    d1 = int(diff/2)
                    d2 = diff - d1
                if stride != 1:
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, d1, d2), "constant", 0))
                else:
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x, (0, 0, 0, 0, d1, d2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet_Light(nn.Module):
    def __init__(self, block, num_blocks, num_filters, num_classes=10, resolution=32, image_channels=3, batchnorm=True, init_normal=True):
        super(ResNet_Light, self).__init__()
        self.in_planes = num_filters[0]

        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(block, num_filters[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_filters[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], num_blocks[2], stride=2)
        self.linear = nn.Linear(num_filters[3][-1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        ii = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes[ii], stride))
            self.in_planes = planes[ii] * block.expansion
            ii += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # print(out.size())
        out = self.layer1(out)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = F.avg_pool2d(out, out.size()[3])
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        # print(out.size())
        return out


if __name__ == "__main__":
    vgg = VGG(Conv_bn, [1,1,1,1,1], [[8], [8], [8], [8], [8], [100,100]])
    print(vgg)
    a = torch.FloatTensor(64,3,224,224)
    print(vgg(a))

    model = resnet20()
    print(model)
    a = torch.randn(1,3,32,32)
    model(a)





