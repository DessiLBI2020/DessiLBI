from __future__ import print_function

import argparse
import math
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.misc import get_conv_zero_param
from mask_utils import analysis_masks


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='results', type=str, metavar='PATH',
                    help='path to the initialization checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# add for load gamma supp
parser.add_argument('--gamma_epoch', default='1', type=str,help='load gamma supp from epoch')
parser.add_argument('--gamma_supp', default="True", type=str,help='if use gamma supp')
parser.add_argument('--percent', default=0.6, type=float)
parser.add_argument('--sgd_epoch', default=-1, type=int)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 100000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    if args.gamma_supp=="True":
        writer = SummaryWriter("{}/tblogs/gamma_epoch{}".format(args.save_dir,args.gamma_epoch))
    else:
        writer = SummaryWriter("{}/tblogs/norm_prune_sgdepoch{}_p{}".format(args.save_dir,args.sgd_epoch,args.percent))
    global best_acc
    start_epoch = args.start_epoch  
    
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    if True :
        print('==> Preparing dataset %s' % args.dataset)
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
            dataloader = datasets.CIFAR10
            num_classes = 10
        else:
            dataloader = datasets.CIFAR100
            num_classes = 100
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    if True:
        print("==> creating model '{}'".format(args.arch))
        if args.arch.endswith('resnet'):
            model = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                    )
            model_ref = models.__dict__[args.arch](
                        num_classes=num_classes,
                        depth=args.depth,
                    )
        else:
            model = models.__dict__[args.arch](num_classes=num_classes)
            model_ref = models.__dict__[args.arch](num_classes=num_classes)

        model.cuda()
        model_ref.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 

    # Resume
    title = 'cifar-10-' + args.arch
    if args.save_dir:
        # Load checkpoint.
        if args.gamma_supp=="True":
            print('==> Getting reference model from gamma_supp..')
            assert os.path.isfile("{}/masks/epoch{}.t7".format(args.save_dir,args.gamma_epoch)), 'Error: no checkpoint directory found!'
            mask_dict = torch.load("{}/masks/epoch{}.t7".format(args.save_dir,args.gamma_epoch))
            best_acc = 0.0 # TODO: check it 
            start_epoch = args.start_epoch
            for name, param in model_ref.named_parameters():
                if name in mask_dict:
                    mask = mask_dict[name].cuda()
                    param.data.mul_(mask)
            res_str,res_list=analysis_masks(mask_dict)
            writer.add_text("mask_analysis",res_str)
            for i in range(len(res_list)):
                writer.add_scalar("layer sparse rate",res_list[i],global_step=i)
        else:
            print('==> Getting reference model from weight..')
            if args.sgd_epoch==-1:
                print("Using SGD best weights!")
                checkpoint = torch.load("{}/pruned_p{}.pth.tar".format(args.save_dir,args.percent))
            else:
                print("Using SGD weight prune @ epoch {}".format(args.sgd_epoch))
                checkpoint = torch.load("{}/pruned_p{}_epoch{}.pth.tar".format(args.save_dir,args.percent,args.sgd_epoch))
            best_acc = checkpoint['best_acc']
            start_epoch = args.start_epoch
            model_ref.load_state_dict(checkpoint['state_dict'])

    logger = Logger(os.path.join(args.save_dir, str(args.percent) + '_' + str(args.sgd_epoch) + '_log_scratch.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # set some weights to zero, according to model_ref ---------------------------------
    assert args.model
    if args.save_dir:
        print('==> Loading init model from %s'%args.model)
        checkpoint = torch.load("{}/init.pth.tar".format(args.save_dir))
        model.load_state_dict(checkpoint['state_dict'])
    # setting zeros
    for m, m_ref in zip(model.modules(), model_ref.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m_ref.weight.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m.weight.data.mul_(mask)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        num_zeros = get_conv_zero_param(model)
        print('Zero parameters: {}'.format(num_zeros))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))
        print("Prune rate: {}".format(float(num_zeros)/num_parameters))
        writer.add_scalar("Prune rate",float(num_zeros)/num_parameters)
        writer.add_scalar("Parameters",num_parameters)
        writer.add_scalar("Zero parameters",num_zeros)

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        writer.add_scalar("train_loss",train_loss,global_step=epoch)
        writer.add_scalar("train_err",100-train_acc,global_step=epoch)
        writer.add_scalar("test_loss",test_loss,global_step=epoch)
        writer.add_scalar("test_err",100-test_acc,global_step=epoch)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir)

    logger.close()

    print('Best acc:')
    print(best_acc)
    writer.add_scalar("Best_err",100-best_acc)



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint, filename='scratch.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
