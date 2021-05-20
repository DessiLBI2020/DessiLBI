'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def descent_lr_train(lr, epoch, optimizer, interval):
    for k in interval:
        if epoch < k:
            break
        else:
            lr /= 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('***********************************')
    print('learning rate:', lr)
    print('***********************************')


def generate_mask(weight_dict, ratio):
    print('Generate Mask')
    ratio =  (1- ratio) 
    mask = OrderedDict()
    for  i, name in enumerate(weight_dict):
        p = weight_dict[name]
        if len(p.size()) == 2:
            size =  p.size()
            length =  p.size()[0] * p.size()[1]
            #thre_index = int(ratio[name] * length)
            #print(thre_index)
            p_numpy = torch.abs(p.data).view(-1).cpu().numpy()
            #p_numpy = np.sort(p_numpy)
            thre = np.quantile(p_numpy, ratio)
            mask[name] = torch.gt(torch.abs(p.data), thre).float().cuda()
            mask[name] = mask[name].view(size[0], size[1])
        elif len(p.size()) == 4:
            size =  p.size()
            length =  p.size()[0] * p.size()[1] * p.size()[2] * p.size()[3]
            p_reshape = torch.reshape(p,(p.shape[0],-1))
            p_norm = torch.norm(p_reshape,2,1)
         #   print(p_norm.size())
         #   ssss
            p_numpy = torch.abs(p_norm).view(-1).cpu().numpy()
            thre = np.quantile(p_numpy, ratio)
   #         print(p_numpy)
  #          print(thre)
            prune_index = [i for i in range(len(p_numpy)) if p_numpy[i] < thre]
 #           print(prune_index)
            mask[name] = torch.zeros_like(p.data).float().cuda() + 1
            mask[name][prune_index] =  0
#            print(mask[name])

    return mask
