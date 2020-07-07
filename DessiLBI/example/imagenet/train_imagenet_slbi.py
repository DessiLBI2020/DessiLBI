# This is the file for training slbi with Imagenet
import os
import torch
from slbi_toolbox import SLBI_ToolBox
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import model.resnet as resnet
import model.alexnet as alexnet
from torchvision import transforms
from utils.train_val_utils import *
from data_loader import load_data
#######   Turn on cudnn
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5, 7 '
parser = argparse.ArgumentParser()
########## SGD initialize
parser.add_argument("--lr_initial", default=1e-1, type=float)  # initial learning rate
parser.add_argument("--mu", default=1000, type=float)  # initial learning rate
parser.add_argument("--kappa", default=1, type=float)  # initial learning rate
parser.add_argument("--momentum", default=0.9, type=float)  # momentum
parser.add_argument("--weight_decay", default=1e-4, type=float)  # weight decay
parser.add_argument("--nesterov", default=False, type=str2bool)   # nesterov
parser.add_argument("--slbi_mode", default=1, type=int)
'''
1 for all add bregman
2 for only weights
3 for only non-bn weigts
'''
########## Training settings
parser.add_argument("--model_name", default='res18', type=str)
parser.add_argument("--train_batch_size", default=256, type=int)
parser.add_argument("--val_batch_size", default=64, type=int)
parser.add_argument("--learning_rate_interval", default=30, type=int)
parser.add_argument("--print_interval", default=100, type=int)
parser.add_argument("--checkpoint_interval", default=1, type=int)
parser.add_argument("--record_interval", default=100, type=int)
parser.add_argument("--weights_interval", default=1, type=int)
parser.add_argument("--train_from_scratch", default=True, type=str2bool) # whether train from scratch
parser.add_argument("--mix_epoch", default=60, type=int) 
parser.add_argument("--mix_training", default=False, type=str2bool) 

parser.add_argument("--load_from_checkpoint", default=False, type=str2bool) # whether continue training
parser.add_argument("--pretrained_model_path", type=str)
parser.add_argument("--previous_checkpoint_path", type=str)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=True, type=str2bool)
parser.add_argument("--epoch_start", default=1, type=int)
parser.add_argument("--epoch_end", default=100, type=int)
parser.add_argument("--gpu_num", default='1', type=str)
parser.add_argument("--all_gpu_num", default='2,3, 4, 5', type=str)
########### Output Path
parser.add_argument("--save_check_point", default=False, type=str2bool)
parser.add_argument("--check_point_name", type=str)
parser.add_argument("--check_point_path", type=str)
parser.add_argument("--save_weights", default=False, type=str2bool)
parser.add_argument("--weights_name", type=str)
parser.add_argument("--weights_name_path", type=str)
parser.add_argument("--gamma_name", type=str)
parser.add_argument("--gamma_name_path", type=str)
parser.add_argument("--record_path", type=str)

args = parser.parse_args()


######## Choose whether to use Cuda
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
torch.cuda.empty_cache()

####### Pick according model
if args.model_name == 'res50':
    model = resnet.resnet50().to(device)
elif args.model_name == 'res101':
    model = resnet.resnet101().to(device)
elif args.model_name == 'res152':
    model = resnet.resnet152().to(device)
elif args.model_name == 'res34':
    model = resnet.resnet34().to(device)
elif args.model_name == 'res18':
    model = resnet.resnet18().to(device)
elif args.model_name == 'alexnet':
    model = alexnet.alexnet().to(device)
else:
    print('Wrong Model Name')

########### Whether to parallel the model
if args.use_cuda:
    if args.parallel:
        model = nn.DataParallel(model)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

############## Whether to load pretrained model
#if args.train_from_scratch:
#    pass
#else:
#    model.load_state_dict(torch.load(args.pretrained_model_path))

# initialize the Optimizer
optimizer = SLBI_ToolBox(model.parameters(), lr=args.lr_initial, momentum=args.momentum, mu=args.mu, kappa=args.kappa,weight_decay=args.weight_decay, nesterov=args.nesterov)

######################### Whether to continue training
#if args.load_from_checkpoint:
#    checkpoint_dict = torch.load(args.previous_checkpoint_path)
#    model.load_state_dict(checkpoint_dict['model'])
#    optimizer.load_state_dict(checkpoint_dict['optimizer'])
##else:
#    pass


name_list = []
name_list_2 = []
name_list_3 = []
#name_file = open(txt_path, 'w')
for name, p in model.named_parameters():
    name_list.append(name)
   # name_file.write(name)
   # name_file.write('\n')
    if 'bias' not in name:
        name_list_2.append(name)
        if 'bn' not in name:
            name_list_3.append(name)
    print(name)
    print(p.size())
#name_file.close()


######## Add name list to the optimizer
optimizer.assign_name(name_list)
optimizer.initialize_slbi(name_list_3)


train_loader = load_data(dataset="ImageNet", train=True, batch_size=args.train_batch_size, shuffle=True)
test_loader = load_data(dataset="ImageNet", train=False, batch_size=args.val_batch_size, shuffle=False)
all_num = args.epoch_end * len(train_loader)

print('num of all step:', all_num)
print('num of step per epoch:', len(train_loader))
print("Test Model")
evaluate_batch(model, test_loader, device)
for ep in range(args.epoch_start,args.epoch_end+1):
    model.train()

    # descend the learning rate
    descent_lr(args.lr_initial, ep, optimizer, args.learning_rate_interval)
    loss_val = 0
    correct = num = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure=None, record=False, path=args.record_path)
        _, pred = logits.max(1)
        loss_val += loss.item()
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
        if (iter + 1) % args.print_interval == 0:
            print('*******************************')
            print('epoch : ', ep)
            print('iteration : ', iter + 1)
            print('loss : ', loss_val/100)
            print('Correct : ', correct)
            print('Num : ', num)
            print('Train ACC : ', correct/num)
            correct = num = 0
            loss_val = 0
    print('Test Model')
    evaluate_batch(model, test_loader, device)






