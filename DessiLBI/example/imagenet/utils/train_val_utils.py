import torch
import numpy as np
from collections import OrderedDict


def str2bool(value):
	# This is used for argparse
    return value.lower() == 'true'


def evaluate_batch(model, data_loader, device):
	# This function is used to evaluate the results on Validation Set
	model.eval()
	correct = num = correct_t5 =0
	for iter, pack in enumerate(data_loader):
		data, target = pack[0].to(device), pack[1].to(device)
		logits = model(data)
		_, pred = logits.max(1)
		_, pred_t5 = torch.topk(logits, 5, dim=1)
		correct += pred.eq(target).sum().item()
		correct_t5 += pred_t5.eq(torch.unsqueeze(target, 1).repeat(1, 5)).sum().item()
		num += data.shape[0]
	print('Correct : ', correct)
	print('Num : ', num)
	print('Test ACC : ', correct / num)
	print('Top 5 ACC : ', correct_t5 / num)
	torch.cuda.empty_cache()
	model.train()
	return correct / num


def save_checkpoints(model, optimizer, path):
	# Save both model weights and optimizer state as .pth
	save_dict = {'model': model.state_dict(), 'optimizer':optimizer.state_dict()}
	torch.save(save_dict, path)

def save_model(model, path):
	# Save both model weights and optimizer state as .pth
	torch.save(model.state_dict(), path)

def descent_lr(lr, epoch, optimizer, interval):
	# desecent the learning rate by interval 
	lr = lr * (0.1 ** ((epoch-1) //interval))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		print('***********************************')
		print('learning rate:', lr)
		print('***********************************')
