import torch
import numpy as np
from collections import OrderedDict

def str2bool(value):
    return value.lower() == 'true'

def train_single_gpu(model, train_loader, optimizer,device):
	pass

def evaluate_batch(model, data_loader, device):
	model.eval()
	correct = num = 0
	for iter, pack in enumerate(data_loader):
		data, target = pack[0].to(device), pack[1].to(device)
		logits = model(data)
		_, pred = logits.max(1)
		correct += pred.eq(target).sum().item()
		num += data.shape[0]
	print('Correct : ', correct)
	print('Num : ', num)
	print('Test ACC : ', correct / num)
	torch.cuda.empty_cache()
	model.train()
	return correct/num

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
        for i in range(1, len(epoch_schedule)):
            if epoch <= epoch_schedule[i]:
                break
            else:
                flag = i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[flag]
        print('***********************************')
        print('epoch:', epoch+1)
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

def prune_model(model, optimizer, method):
	pass


def permute_input(img_tensor, patch_size, permute_rate):
	batch_size, channel, width, height = img_tensor.size()
	assert(width == height)
	n_d = int(width / patch_size)
	n_permute = int(permute_rate * n_d)
        index_array = np.array(range(n_d))
	pick_array = np.random.choice(n_d, n_permute)
	permutation = np.random.permutation(n_permute)
	for i in range(n_permute):
		index_array[pick_array[i]] = pick_array[permutation[i]]
	permute_matrix = np.zeros((width, height))
	for i in range(n_d):
		for j in range(n_d):
			i_r = range(index_array[i]*patch_size, (index_array[i]+1)*patch_size)
			j_r = range(index_array[j]*patch_size, (index_array[j]+1)*patch_size)
			permute_matrix[i*patch_size : (i+1)*patch_size, j*patch_size : (j+1)*patch_size] = np.meshgrid( i_r , j_r)
			

