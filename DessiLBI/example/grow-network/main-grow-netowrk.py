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

from utils import get_model, load_data, adjust_learning_rate, layer_name2id, get_new_arc, grow_filter,  get_name_list, get_thresh_hold, train, test, plot_training_curve
from slbi_toolbox import SLBI_ToolBox
from flops_counter import get_model_complexity_info



parser = argparse.ArgumentParser(description='DessiLBI growing networks')
parser.add_argument('--network', default='vgg', type=str, help='resnet, VGG')
parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10, cifar100, imagenet')
parser.add_argument('--model', default='vgg16', type=str, help='resnet32, vgg16') 
parser.add_argument('--batchsize',default=128,type=int, help='batch size')
parser.add_argument('--local_epoch', default=10, type=int, help='')
parser.add_argument('--lr', default=0.05, type=float, help='')
parser.add_argument('--mu', default=100, type=int, help='mu for slbi')
parser.add_argument('--kappa', default=1, type=int, help='kappa for slbi')
parser.add_argument('--finetune_epoch', default=40, type=int, help='finetuning epoch')
parser.add_argument('--tau',default=0.8, type=float, help='the grow threshold for growing filters')
parser.add_argument('--mix_tau', default=False, type=bool, help='')
parser.add_argument('--grow_ratio', default=1.1, type=float, help='ratio for growing filter if filter_grow==ratio')
parser.add_argument('--grow_num',default=4,type=int,help='')
parser.add_argument('--weight_decay',default=0.0,type=float,help='')

args = parser.parse_args()

#preparing log
timestamp = str(datetime.datetime.now()).split('.')[0]
if not os.path.exists('./output'):
    os.mkdir('./output')  
save_path = './output/' + args.model +'-'+ args.dataset
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_path = os.path.join(save_path, timestamp)
if not os.path.exists(save_path):
    os.mkdir(save_path)

logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
logger = logging.getLogger('main')
logger.addHandler(logging.StreamHandler())
logger.info("Saving to %s", save_path)

#print parameters
logger.info('--------args----------')
for k in list(vars(args).keys()):
    logger.info('%s: %s' % (k, vars(args)[k]))
logger.info('--------args----------\n')


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#prepare model
model, BLOCK, BLOCK_NUM, FILTER_NUM, NUM_CLASSES, RESOLUTION, NET = get_model(args)
model.to(device) 
logger.info('model: \n{}'.format(model))

criterion = nn.CrossEntropyLoss().to(device)

#initialize optimizer
optimizer = SLBI_ToolBox(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu,weight_decay=args.weight_decay)
layer_list, slbi_layer_list = get_name_list(model)
optimizer.assign_name(layer_list)
optimizer.initialize_slbi(slbi_layer_list)

#prepare dataset
trainloader, testloader = load_data(logger, args)



G_FILTER = True
G_LAYER = False

BEST_ACC = 0
train_acc = [0.0]
test_acc = [0.0]
loss_list = []

MAC_list = []
params_list = []

iteration = 1
epoch = 0

start_time = time.time()
#grow network
while True:  
    logger.info('Iteration **{}**'.format(iteration))
    mac, params = get_model_complexity_info(model, (3, RESOLUTION, RESOLUTION), as_strings=True, print_per_layer_stat=False)
    MAC_list.append(mac)
    params_list.append(params)

    GROWN = False
       
    tr_acc, loss = train(logger, epoch, model, trainloader, criterion, optimizer, device)
    te_acc = test(logger, epoch, model, testloader, criterion, device)

    train_acc.append(tr_acc)
    test_acc.append(te_acc)
    loss_list.append(loss)

    if te_acc > BEST_ACC:    #save the best checkpoint
        BEST_ACC = te_acc
        logger.info('Saving best %.3f @ %d ...' %(te_acc, epoch))
        torch.save(model.state_dict(), os.path.join(save_path, 'best_ckpt.t7'))
        with open(os.path.join(save_path, 'best_ckpt.txt'), 'w') as file:
            file.write('BLOCK_NUM: ' + str(BLOCK_NUM) + '\n')
            file.write('FILTER_NUM: ' + str(FILTER_NUM) + '\n')
            file.write('best_acc: ' + str(BEST_ACC)) 

    #check whether to grow in every args.local_epoch
    if epoch == args.local_epoch-1:
        for conv_layer in slbi_layer_list:
            if conv_layer in ['linear.weight','linear3.weight']  or 'shortcut' in conv_layer:
                continue
            select_rate= optimizer.calculate_proportion(conv_layer)
            if select_rate >= get_thresh_hold(conv_layer, args):    #growing filters
                    GROWN = True
                    epoch = 0  
                    logger.info('Current growing layer is {}'.format(conv_layer))
                    G_LAYER = False; G_FILTER = True
                    get_new_arc(logger, args, current_layer=conv_layer, G_FILTER=G_FILTER, G_LAYER=G_LAYER, FILTER_NUM=FILTER_NUM, BLOCK_NUM=BLOCK_NUM )
        if GROWN:
            model = grow_filter(model, [BLOCK, BLOCK_NUM, FILTER_NUM, NUM_CLASSES, RESOLUTION], NET, args,logger)
            model.to(device)
                
            optimizer = SLBI_ToolBox(model.parameters(), lr=args.lr, kappa=args.kappa, mu=args.mu, weight_decay=args.weight_decay)
            logger.info('the new model is : \n {} '.format(model))
            layer_list, slbi_layer_list = get_name_list(model)
            optimizer.assign_name(layer_list)
            optimizer.initialize_slbi(slbi_layer_list)
            
            iteration += 1
        else:
            break
    else:
        epoch += 1
        
growing_time = time.time()-start_time

for epoch in range(args.finetune_epoch):
    adjust_learning_rate(optimizer, logger, args, epoch)
    tr_acc, loss = train(logger, epoch, model, trainloader, criterion, optimizer, device)
    te_acc = test(logger, epoch, model, testloader, criterion, device)  
    train_acc.append(tr_acc)
    test_acc.append(te_acc)
    loss_list.append(loss)
    if te_acc > BEST_ACC:
        BEST_ACC = te_acc
        logger.info('Saving best epoch {}...'.format(len(train_acc)))
        state = {'model': model.state_dict(),
                 'epoch': len(train_acc),
                 'best_acc': BEST_ACC}
        torch.save(state, os.path.join(save_path, 'finetune_best_ckpt.t7'))

total_time = time.time() - start_time
mac, params = get_model_complexity_info(model, (3, RESOLUTION, RESOLUTION), as_strings=True, print_per_layer_stat=False)
MAC_list.append(mac)
params_list.append(params)

logger.info('train_acc: {}'.format(train_acc))
logger.info('test_acc: {}'.format(test_acc))
logger.info('best test acc : {}'.format(max(test_acc)))
logger.info('mean acc of last 5 epoch : {}'.format(sum(test_acc[-5:])/5.0))
logger.info('mac:{}'.format(MAC_list))
logger.info('params:{}'.format(params_list))
logger.info('growing time: {}, total time: {}'.format(growing_time, total_time))
logger.info('Final model is {}'.format(model))

#plot and save training curve
plot_training_curve(train_acc, test_acc, loss_list, save_path)


