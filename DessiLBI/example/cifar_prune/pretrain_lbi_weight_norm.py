'''Train CIFAR11 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import argparse
from model.MobileNetV2  import MobileNetV2_CIFAR10
from model.resnet_cifar10 import *
from model.vgg import *
from train_utils import progress_bar, descent_lr_train
from slbi_noconv_with_normalize import SLBI
from model.resnet import *
os.makedirs('./txtrecorder/', exist_ok=True)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--kappa', type=float, default=1)
parser.add_argument('--mu', type=float, default=1000)
parser.add_argument('--lbi_momentum', type=float, default=0.5)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--post_mode', type=str, default='finetune', help="finetune or retrain") 
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--fc_lambda', type=float, default=0.1,
                        help='Slbi fc_lambda (default: 1)')

parser.add_argument('--conv_lambda', type=float, default=0.1,
                        help='Slbi fc_lambda (default: 1)')
parser.add_argument('--logname', type=str, default='sgd_mobile', help="name \
                        of save log") 
parser.add_argument('--model', type=str, default='resnet20', help="name \
                        of save log") 
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
writer = SummaryWriter(args.logname)
print(args.logname.split('/')[-1] + '.pth')
# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
# Model
print('==> Building model..')
if args.model == 'mobilev2':
    net = MobileNetV2_CIFAR10()
elif args.model == 'resnet20':
    net = ResNet20()
elif args.model == 'resnet32':
    net = ResNet32()
elif args.model == 'resnet44':
    net = ResNet44()
elif args.model == 'resnet56':
    net = ResNet56()
elif args.model == 'vgg16':
    net = vgg16()
elif args.model == 'vgg16_bn':
    net = vgg16_bn()
elif args.model == 'vgg19':
    net = vgg19()
elif args.model == 'vgg19_bn':
    net = vgg19_bn()
elif args.model == 'resnet50':
    net = resnet(num_classes=10,depth=50)
net = net.to(device)
#if device == 'cuda':
   # net = torch.nn.DataParallel(net)
cudnn.benchmark = True
#print(net.state_dict().keys())
#sss
def write_hist(weight_dict, epoch, name):
    for i, w_name in enumerate(weight_dict):
        if len(weight_dict[w_name].size()) == 2 or len(weight_dict[w_name].size()) == 4:
            weightnpy =  weight_dict[w_name].detach().cpu().numpy().ravel()
            writer.add_histogram(name + '/' +  w_name, weightnpy, epoch)
criterion = nn.CrossEntropyLoss()
optimizer = SLBI(net.parameters(), lr=1e-1, kappa=args.kappa, mu=args.mu, momentum=0.9, weight_decay=5e-4, lbi_momentum=args.lbi_momentum,fc_lambda=args.fc_lambda)
name_list = []
layer_list = []
for name, p in net.named_parameters():
    name_list.append(name)
    print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

initial_dict = copy.deepcopy(net.state_dict())
torch.save(initial_dict, args.logname.split('/')[-1] + '_init.pth')
file_tmp = open('./txtrecorder/' + args.logname.split('/')[-1] + '.txt', 'w')
file_final = open('./txtrecorder/final' + args.logname.split('/')[-1] + '.txt', 'w')
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train Loss : ', train_loss /len(trainloader))

 #       progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
  #                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

   #         progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    print('Test Loss : ', test_loss/len(testloader))
    print('Test Acc : ', acc)
    return acc


def finetune(epoch, optimizer):
    print('Finetune')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step_with_mask()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
     #   progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #               % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
bestacc = 0
bestsacc = 0
for epoch in range(start_epoch, start_epoch+args.epoch):
    descent_lr_train(1e-1, epoch, optimizer, [60, 120, 160])
    train(epoch)
    acc = test(epoch)
    file_tmp.write('Epoch : ')
    file_tmp.write(str(epoch))
    file_tmp.write('\n')
    file_tmp.write('Test Acc : ')
    file_tmp.write(str(acc))
    file_tmp.write('\n')
    file_tmp.write('\n')
    file_tmp.write('\n')
    file_tmp.write('\n')
    file_tmp.flush()
    writer.add_scalar('Dense/Test Acc', acc, epoch)
    if (epoch+1) in [60, 120, 160]:
       torch.save(net.state_dict(),  args.logname.split('/')[-1] + str(epoch)+'_pretrain_w.pth')
       z_dict = optimizer.get_z_state_dict()
       torch.save(z_dict, args.logname.split('/')[-1] + str(epoch) + '_pretrain_z.pth')
       mask, sparsity = optimizer.get_mask()
       torch.save(mask, args.logname.split('/')[-1] + str(epoch) + '_mask.pth')
    optimizer.use_w_star(writer=file_tmp)
    optimizer.check_sparsity(recorder=file_tmp)
    sparse_acc = test(epoch)
    file_tmp.write('\n')
    file_tmp.write('Sparse Test Acc : ')
    file_tmp.write(str(sparse_acc))
    file_tmp.write('\n')
    file_tmp.write('\n')
    file_tmp.write('\n')
    file_tmp.write('\n')
    file_tmp.flush()
    writer.add_scalar('Sparse/Test Acc', sparse_acc, epoch)
    optimizer.recover()
    _, sparsity  = optimizer.get_mask()
    optimizer.check_sparsity(recorder=file_tmp)
    writer.add_scalar('Sparse/Sparsity', sparsity, epoch)
    if bestacc < acc:
        bestacc =  acc
    if bestsacc < sparse_acc:
        bestsacc =  sparse_acc


file_tmp.write('Dense Best Acc : ')
file_tmp.write(str(bestacc))
file_tmp.write('\n')
file_tmp.flush()
file_tmp.write('Sparse Best Acc : ')
file_tmp.write(str(bestsacc))
file_tmp.write('\n')
file_tmp.flush()
writer.add_scalar('Dense/Best Test Acc', bestacc, 0)
writer.add_scalar('Sparse/Best Test Acc', bestsacc, 0)

torch.save(net.state_dict(),  args.logname.split('/')[-1] + 'pretrain_w.pth')
z_dict = optimizer.get_z_state_dict()
torch.save(z_dict, args.logname.split('/')[-1] + 'pretrain_z.pth')

#### get mask
mask, sparsity = optimizer.get_mask()
if args.post_mode == 'retrain':
    print('reload initial dict')
    net.load_state_dict(initial_dict)
optimizer.use_w_star(writer=file_final)
file_final.close()
# writer.add_scalar('sparsity', sparsity, 100)
from sgd_with_mask import SGD
optimizer_new = SGD(net.parameters(), lr=1e-2,  momentum=0.9, weight_decay=5e-4)
name_list = []
layer_list = []
for name, p in net.named_parameters():
    name_list.append(name)
  #  print(name)
    if len(p.data.size()) == 4:
        layer_list.append(name)
optimizer_new.assign_name(name_list)
optimizer_new.load_mask(mask)
optimizer_new.apply_mask()
optimizer_new.check_sparsity(recorder=file_tmp)
print('Before Finetuning')
acc = test(epoch)
file_tmp.write('Acc Before Pruning : ')
file_tmp.write(str(acc))
file_tmp.write('\n')
file_tmp.write('\n')
file_tmp.flush()
##### finetune
bestacc = 0
if args.post_mode == 'finetune':
    print('Finetuning')
    for epoch in range(400):
        descent_lr_train(1e-2, epoch, optimizer_new, [120, 240, 320])
        finetune(epoch, optimizer_new)
        acc = test(epoch)
        optimizer_new.check_sparsity(recorder=file_tmp)
        file_tmp.write('Finetuning Epoch : ')
        file_tmp.write(str(epoch))
        file_tmp.write('\n')
        file_tmp.write('Finetuning Acc : ')
        file_tmp.write(str(acc))
        file_tmp.write('\n')
        file_tmp.flush()
        writer.add_scalar('Finetune/Test Acc', acc, epoch)
        if bestacc < acc:
            bestacc =  acc
    file_tmp.write('Finetune Best Acc : ')
    file_tmp.write(str(bestacc))
    file_tmp.write('\n')
    file_tmp.flush()
    writer.add_scalar('Finetune/Best Test Acc', bestacc, 0)
    torch.save(net.state_dict(),  args.logname.split('/')[-1] + 'final.pth')
    writer.close()
    file_tmp.close()

elif args.post_mode == 'retrain':
    print('Retraining')
    for epoch in range(200):
        descent_lr_train(1e-1, epoch, optimizer_new, [60, 120, 160])
        finetune(epoch, optimizer_new)
        acc = test(epoch)
        optimizer_new.check_sparsity(recorder=file_tmp)
        file_tmp.write('Finetuning Epoch : ')
        file_tmp.write(str(epoch))
        file_tmp.write('\n')
        file_tmp.write('Finetuning Acc : ')
        file_tmp.write(str(acc))
        file_tmp.write('\n')
        file_tmp.flush()
        writer.add_scalar('Finetune/Test Acc', acc, epoch)
        if bestacc < acc:
            bestacc =  acc
    file_tmp.write('Finetune Best Acc : ')
    file_tmp.write(str(bestacc))
    file_tmp.write('\n')
    file_tmp.flush()
    writer.add_scalar('Finetune/Best Test Acc', bestacc, 0)
    torch.save(net.state_dict(),  args.logname.split('/')[-1] + 'final.pth')
    writer.close()
    file_tmp.close()
