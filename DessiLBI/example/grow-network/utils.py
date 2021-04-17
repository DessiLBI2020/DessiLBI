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

import argparse
import copy



def load_data(logger, args):
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        logger.info('[*] Preparing CIFAR10...')
        trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                                download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                               download=True,
                                               transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=8)

    elif args.dataset == 'cifar100':
        logger.info('[*] Preparing CIFAR100...')
        trainset = torchvision.datasets.CIFAR100(root='./', train=True,
                                                download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./', train=False,
                                               download=True,
                                               transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=8)

    else:
        raise ValueError('wrong dataset !')        

    return trainloader, testloader




def adjust_learning_rate(optimizer, logger, args, epoch):
    n = int(args.finetune_epoch/3)

    lr = args.lr * (args.lr ** (epoch // n))
    if lr!= args.lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            logger.info('adjust learning rate to {}'.format(lr))
    
    return 0



def layer_name2id(layer_name):
    if layer_name.split('.')[0] == 'module':
        temp = layer_name.split('.')[1:]
    else:
        temp = layer_name.split('.')

    if temp[0] == 'conv1':
        id = [0]
    elif temp[0] == 'conv2': # for mobilenet
        id = [8]
    elif temp[0][:-1] == 'layer' and temp[2] != 'shortcut':
        if 'conv_pw_1' in temp:
            id = []
        else:
            id = [int(temp[0][-1]), int(temp[1])]
    elif temp[0][:-1] == 'linear':
        id = [int(temp[0][-1]), 0, 0]
    else:
        id = []
    return id



#
def get_new_arc(logger, args, G_FILTER, G_LAYER, FILTER_NUM, BLOCK_NUM, current_layer=None, grow_module=None, filter_grow_num=10, layer_grow_num=1):
    
    ratio = args.grow_ratio
    num = args.grow_num
    if G_FILTER and current_layer is not None:  #grow filter
        layer_id = layer_name2id(current_layer)
        logger.info('layer id: {}'.format(layer_id))
        if len(layer_id) == 1:
            idx = layer_id[0]
            if args.filter_grow == 'ratio':
                FILTER_NUM[idx] = max(FILTER_NUM[idx]+4, int(ratio * FILTER_NUM[idx]))
                    
        elif len(layer_id) == 2:
            if args.network == 'vgg':
                FILTER_NUM[layer_id[0]-1][layer_id[1]] = max(FILTER_NUM[layer_id[0]-1][layer_id[1]]+8, int(ratio * FILTER_NUM[layer_id[0]-1][layer_id[1]]))
            elif args.network in ['resnet','resnet_light']:
                FILTER_NUM[layer_id[0]][layer_id[1]] = max(FILTER_NUM[layer_id[0]][layer_id[1]] + num, int(ratio * FILTER_NUM[layer_id[0]][layer_id[1]]))
            else:
                pass

        elif len(layer_id) == 3:
            if args.network == 'vgg':
                FILTER_NUM[-1][layer_id[0]-1] = 2*FILTER_NUM[-1][layer_id[0]-1]
            elif args.network in ['resnet','resnet_light']:
                logger.info('Linear layer donot need to grow!')
            else:
                raise ValueError('invalid args.network!')
        logger.info('After growing filters, the architecture is blocks:{}****** filters:{}'.format(BLOCK_NUM, FILTER_NUM))
    else:
        logger.info('Failing to get new arc!')


        
def grow_filter(model, new_arc, NET, args, logger, topk_dict=None):
    # new_arc: [basic_block, [block_num list], [filter_num list]]
    # layer_name: the layer to be growed
    old_params = {}
    for n, p in model.named_parameters():
        if 'module' in n:
            n = '.'.join(n.split('.')[1:])
        old_params[n] = p.data

    new_net = NET(new_arc[0], new_arc[1], new_arc[2], num_classes=new_arc[3], resolution=new_arc[4])
    
    for n, p in new_net.named_parameters():
        if n in old_params.keys():
            if p.data.size() != old_params[n].size(): #this layer grown
                old_size = old_params[n].size()
                if len(old_size) == 4:
                    try:  
                        filter_idx = topk_dict[n]
                        n_out, n_in, k1, k2 = old_size
                        for idx in filter_idx:
                            p.data[idx, :n_in, :k1, :k2] = old_params[idx, :, :, :]
                    except: #shortcut weight
                        n_out, n_in, k1, k2 = old_size
                        p.data[:n_out, :n_in, :k1, :k2] = old_params[n]

                elif len(old_size) == 2:
                    num_out, num_in = old_size
                    p.data[:num_out, :num_in] = old_params[n]
                elif len(old_size) == 1:
                    a, = old_size           
                    p.data[:a] = old_params[n]
            else:  #this layer did not grow
                p.data = old_params[n]
            #logger.info('{} has succeed parameters from last model!'.format(n))

        else:
            pass
    
    return new_net



def get_name_list(model):
    layer_list = []
    slbi_layer_list = []
    for n,p in model.named_parameters():
        layer_list.append(n)
        if len(p.data.size()) == 4 :
            if 'shortcut' in n or 'conv_dw_1' in n or 'conv_pw_1' in n:
                continue
            elif 'conv_pw_2' in n:
                slbi_layer_list.append(n)
            elif p.data.size()[-1] == 1:
                continue
            else:
                slbi_layer_list.append(n)
        elif len(p.data.size()) == 2 and n not in ['linear.weight', 'linear3.weight']:
            slbi_layer_list.append(n)
        else:
            pass
    return layer_list, slbi_layer_list


def get_thresh_hold(layer_name, args):
    if args.network == 'resnet_light':
        if args.mix_tau:
            if 'layer1' in layer_name :
                return 0.9
            elif 'layer2' in layer_name:
                return 0.85
            else:
                return 0.8
            
        else:
            return args.tau
    elif args.network == 'vgg' :
        if args.mix_tau:
            if 'layer1.0' in layer_name:
                return 0.6
            elif 'layer2' in layer_name :
                return 0.9
            elif 'layer3' in layer_name:
                return 0.8
                #return 0.9
            elif 'layer4' in layer_name:
                return 0.5
                #return 0.3
            elif 'layer5' in layer_name:
                return 0.3
                #return 0.2
            else:
                return args.tau
        else:
            return args.tau
    


def train( logger, epoch, model, trainloader, criterion, optimizer, device):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        logger = logger)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 2000 == 0:
            progress.display(i)
    
    logger.info(' Training Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.3f}'
              .format(top1=top1, top5=top5, losses=losses))

    return top1.avg.detach().cpu().item(), loss.detach().cpu().item()


def test(logger, epoch, model, testloader,criterion, device):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ',
        logger= logger)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(testloader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' Test Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg.detach().cpu().item()




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res








import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
def plot_training_curve(train_list, test_list=None, loss_list=None, save_path=None):
    plt.figure()
    train_list = [x/100. for x in train_list]
    plt.plot(train_list, c='g', label='train accuracy')
    if test_list is not None:
        test_list = [x/100. for x in test_list]
        plt.plot(test_list, c='r', label='test accuracy')
    # if loss_list is not None:
    #     plt.plot(loss_list, c='b', label='loss')
    
    plt.xlabel('epoch')
    plt.ylabel('accuracy & loss')
    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'training_curve.png'))
    else:
        plt.savefig('training_curve.png')
    plt.close()

    return 0






from model import  ResNet_Light,BasicBlock_Light, VGG, Conv_bn, Conv_3
def get_model(args):
    
    if args.model in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        NET = ResNet_Light
        BLOCK = BasicBlock_Light
    elif args.model in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        NET = VGG
        BLOCK = Conv_bn
    else:
        raise ValueError("wrong model...")
        
    if args.model in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        block_n_dict = {'resnet20':3, 'resnet32':5, 'resnet44':7, 'resnet56':9, 'resnet110':18, 'resnet1202':200}
        BLOCK_NUM = [block_n_dict[args.model]] * 3
        FILTER_NUM = [16, 
                        [16]*block_n_dict[args.model],
                        [16]*block_n_dict[args.model],
                        [16]*block_n_dict[args.model]]

    elif args.model == 'vgg16':
        BLOCK_NUM = [2, 2, 3, 3, 3]
        FILTER_NUM = [[16, 16], 
                        [16, 16], 
                        [16, 16, 16], 
                        [16, 16, 16], 
                        [16, 16, 16], 
                        [512]] 
    elif args.model == 'vgg19':
        BLOCK_NUM = [2, 2, 4, 4, 4]
        FILTER_NUM = [[16, 16],
                        [16, 16], 
                        [16, 16, 16, 16], 
                        [16, 16, 16, 16], 
                        [16, 16, 16, 16], 
                        [512]]
    else:
        raise ValueError('wrong model!')

    #get resolusion and classes
    RESOLUTION = 32
    if args.dataset == 'cifar10':
        NUM_CLASSES = 10
    elif args.dataset == 'cifar100':
        NUM_CLASSES = 100
    else:
        raise ValueError("Wrong dataset ...")
    
    model = NET(BLOCK, BLOCK_NUM, FILTER_NUM, NUM_CLASSES, resolution=RESOLUTION, init_normal=True)

    return model, BLOCK, BLOCK_NUM, FILTER_NUM, NUM_CLASSES, RESOLUTION, NET



