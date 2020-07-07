import torch
import numpy as np
from collections import OrderedDict

def str2bool(value):
    return value.lower() == 'true'

def train_single_gpu(model, train_loader, optimizer,device):
	pass

def evaluate_batch(model, data_loader, device):
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

def evaluate_single(model, data_loader, device):
	model.eval()
	correct = 0
	for iter, pack in enumerate(data_loader):
		data, target = pack[0].to(device), pack[1].to(device)
		logits = model(data)
		pred = logits.max(1, keepdim=True)[1]
		if pred.item() == target.item():
			correct += 1
	print('Test ACC : ', correct / len(data_loader))
	torch.cuda.empty_cache()
	model.train()

def save_model_and_optimizer(model, optimizer, path):
	save_dict = {'model': model.state_dict(), 'optimizer':optimizer.state_dict()}
	torch.save(save_dict, path)



def adjust_learning_rate(star_lr, current_s, max_num, optimizer, expo):
	# this one is used to adjust learning rate, call at the end of epoch
	for group in optimizer.param_groups:
		lr_old = group['lr']
		lr_new = star_lr * (1 - current_s / max_num) ** expo
		group['lr'] = lr_new
	if current_s % 100 == 0:
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
		print('step :', current_s)
		print('new_learning_rate:',lr_new)
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


def assign_lr(lr_new, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_new

def descent_lr_with_warmup(epoch, optimizer, lr_schedule, epoch_schedule):
        position_list = range(len(lr_schedule))
        flag = 0
        for i in range(0, len(epoch_schedule)):
            if epoch < epoch_schedule[i]:
                break
            else:
                flag = i+1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[flag]
        print('***********************************')
        print('epoch:', epoch)
        print('learning rate:', lr_schedule[flag])
        print('***********************************')

def descent_lr(lr, epoch, optimizer, interval):
        lr = lr * (0.1 ** (epoch //interval))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('***********************************')
        print('learning rate:', lr)
        print('***********************************')


def mgpu_dict_to_sgpu_dict(weights_dict):
	w_d = OrderedDict()
	for k, v in weights_dict.items():
    		new_k = k.replace('module.', '')
    		print(new_k)
    		w_d[new_k] = v
	return w_d

def load_model_and_optimizer(model, optimizer, path):
	loaded_dict = torch.load(path)
	model.load_state_dict(loaded_dict['model'])
	optimizer.load_state_dict(loaded_dict['optimizer'])

def prune_model_sgd(model, optimizer, mode, device, layer_name, percent):
	pass

def prune_model_slbi(model, optimizer, test_loader,mode, device, layer_name, percent, prune_bias=True, recover=True, evaluate=True):
	if mode == 'new_prune':
		thre = optimizer.cal_thre(percent, layer_name)
		print('Threshold:', thre)
		optimizer.prune_layer_2(percent=percent, layer_name=layer_name, prune_bias=True)
		if evaluate:
			evaluate_batch(model, test_loader, device)
		if recover:
			optimizer.recover()
			
def process_bn(model_dict):
	bn_list = []
	for i, key in enumerate(model_dict):
		if 'bn.weight' in key:
			bn_list.append(key)
	for i, name in enumerate(bn_list):
		new_size = model_dict[name].size()[0]
		new_running_mean = torch.zeros_like(model_dict[name])
		new_running_var = torch.zeros_like(model_dict[name])
		old_size = model_dict[name.replace('weight', 'running_mean')].size()[0]
		new_running_mean[0:old_size] = model_dict[name.replace('weight', 'running_mean')]
		new_running_var[0:old_size] = model_dict[name.replace('weight', 'running_var')]
		model_dict[name.replace('weight', 'running_mean')] = new_running_mean
		model_dict[name.replace('weight', 'running_var')] = new_running_var


def enlarge_weights_slbi(model, optimizer, test_loader, device, layer_name, enlarge_coefficien, percent, enlarge_bias=True, recover=True, evaluate=True):
	thre = optimizer.cal_thre(percent, layer_name)
	print('Threshold:', thre)
	optimizer.enlarge_weak_filters(enlarge_coefficien, percent, layer_name, enlarge_bias)
	if evaluate:
		evaluate_batch(model, test_loader, device)
	if recover:
		optimizer.recover()
