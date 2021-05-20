#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from torch.optim.optimizer import Optimizer, required
import copy
from collections import OrderedDict

class SLBI(Optimizer):

	def __init__ (self, params, lr=required, kappa=1, mu=100, weight_decay=0, momentum=0.9, dampening=0, lbi_momentum=0, fc_lambda=0.1):
		defaults = dict(lr=lr, kappa=kappa, mu=mu, weight_decay=weight_decay, momentum=momentum, dampening=dampening, lbi_momentum=lbi_momentum, fc_lambda=fc_lambda)
		print('*******************************************')
		for key in defaults:
			print(key, ' : ', defaults[key])
		print('*******************************************')
		super(SLBI, self).__init__(params, defaults)


	def __setstate__(self, state):
		super(SLBI, self).__setstate__(state)


	def assign_name(self, name_list):
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				param_state['name'] = name_list[iter]


	def initialize_slbi(self, layer_list=None):
		if layer_list == None:
			pass
		else:
			for group in self.param_groups:
				for p in group['params']:
					param_state = self.state[p]
					if param_state['name'] in layer_list:
						param_state['z_buffer'] = torch.zeros_like(p.data)
						param_state['gamma_buffer'] = torch.zeros_like(p.data)


	def step(self, closure=None):
        ### add normalization
		loss = None
		if closure is not None:
			loss = closure()
		for group in self.param_groups:
			mu = group['mu']
			kappa = group['kappa']
			lr_kappa = group['lr'] 
			lr_gamma = group['lr'] / (mu * kappa)
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			lbi_momentum = group['lbi_momentum']
			fc_lambda = group['fc_lambda']
			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data
				param_state = self.state[p]

				if momentum != 0:
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
						buf.mul_(momentum).add_(d_p)
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					d_p = buf

				if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
					d_p.add_(weight_decay, p.data)

				if  'z_buffer' in param_state:
					new_grad = d_p * lr_kappa + (p.data - param_state['gamma_buffer']) * lr_kappa / mu 
					last_p = copy.deepcopy(p.data)				
					p.data.add_(-new_grad)
					#### lbi momentum
					gamma_grad = param_state['gamma_buffer'] - last_p
					if lbi_momentum != 0:
						if 'gamma_momentum_buffer' not in param_state:
							gbuf = param_state['gamma_momentum_buffer'] = torch.zeros_like(p.data)
							gbuf.mul_(lbi_momentum).add_(gamma_grad)
						else:
							gbuf = param_state['gamma_momentum_buffer']
							gbuf.mul_(lbi_momentum).add(gamma_grad)
						gamma_grad = gbuf 

					#### use W norm
					tmp_p = p.data.clone()
				#	tmp_p = torch.reshape(tmp_p,(tmp_p.shape[0],-1))
					norm_scale =  torch.abs(tmp_p) 
					norm_scale = torch.clamp(norm_scale, 1) ### do not enlarge grad
					del tmp_p
					######### cal proportion
					tmp_p = param_state['gamma_buffer'].clone()
					gamma_ratio = torch.gt(torch.abs(tmp_p),0).float().sum().item()/tmp_p.size().numel()
					# gamma_ratio = gamma_ratio.to(param_state['gamma_momentum_buffer'].device)
					gamma_proportion =  min((1-gamma_ratio), 1e-2)
					param_state['z_buffer'].add_(-lr_gamma, gamma_grad * gamma_proportion /norm_scale)
					# if len(p.data.size()) == 2:
					# 	param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], 1)
					if len(p.data.size()) == 4 or len(p.data.size()) == 2:
						param_state['gamma_buffer'] = kappa * norm_scale * self.shrink(param_state['z_buffer'], fc_lambda)
					else:
						pass
				else:
					p.data.add_(-lr_kappa, d_p)#for bias update as vanilla sgd


	def calculate_w_star_by_layer(self, layer_name):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if  'z_buffer' in param_state and param_state['name'] == layer_name:
					if len(p.data.size()) == 2:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					elif len(p.data.size()) == 4:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					else:
						pass
				else:
					pass


	def calculate_all_w_star(self):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if  'z_buffer' in param_state:
					#if len(p.data.size()) == 2:
	#					print(p.data.size())
	#					print(param_state['gamma_buffer'].size())
						#param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					if len(p.data.size()) == 4 or len(p.data.size()) == 2:
	#					print(p.data.size())
	#					print(param_state['gamma_buffer'].size())
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					else:
						pass

		

	def calculate_layer_residue(self, layer_name):
		diff = 0
		for group in self.param_groups:
			mu = group['mu']
			for p in group['params']:
				param_state = self.state[p]
				if param_state['name'] == layer_name:
					if 'gamma_buffer' in param_state:
						diff = ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
					else:
						pass
		diff /= (2*mu)
		print('Residue of' + layer_name + ' : ', diff)


	def calculate_all_residue(self):
		diff = 0
		for group in self.param_groups:
			mu = group['mu']
			for p in group['params']:
				param_state = self.state[p]
				if 'gamma_buffer' in param_state:
					diff += ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
		diff /= (2*mu)
		print('Residue : ', diff)


	def shrink(self, s_t, lam):
		#proximal mapping for 2-d weight(fc layer)
		gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
		return gamma_t


	def shrink_group(self, ts):
		# shrinkage for 4-d weight(conv layer)
		ts_reshape = torch.reshape(ts,(ts.shape[0],-1))
		ts_norm = torch.norm(ts_reshape,2,1)
		ts_shrink = torch.max(torch.zeros_like(ts_norm),torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm),ts_norm))
		ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape,0,1),ts_shrink),0,1)
		ts_return = torch.reshape(ts_return,ts.shape)
		return ts_return


	def use_w_star(self, writer=None):
		#use sparse params to replace original params
		self.calculate_all_w_star()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'w_star' in param_state:
					if len(p.data.size())==4 or len(p.data.size())==2:
						N = p.size().numel()
						N_s = torch.gt(torch.abs(param_state['w_star']), 0.0).float().sum().item()
						ts_norm = param_state['w_star']
						z_norm = param_state['z_buffer']
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data = param_state['w_star']
						print(param_state['name'])
						print('max w star:', torch.max(ts_norm))
						print('min w star:', torch.min(ts_norm))
						print('max z norm:', torch.max(z_norm))
						print('min z norm:', torch.min(z_norm))
						print('number of weight: ',  N)
						print('number of selected weight:', N_s)
						if writer:
							writer.write(param_state['name'])
							writer.write('\n')
							writer.write('max w star:')
							writer.write( str(torch.max(ts_norm)))
							writer.write('\n')
							writer.write('min w star:' )
							writer.write( str(torch.min(ts_norm)))
							writer.write('\n')
							writer.write('max z norm:')
							writer.write( str(torch.max(z_norm)))
							writer.write('\n')
							writer.write('min z norm:')
							writer.write( str(torch.min(z_norm)))
							writer.write('\n')
							writer.write('number of weight: ' )
							writer.write(str(N) )
							writer.write('\n')
							writer.write('number of selected weight:')
							writer.write( str(N_s) )
							writer.write('\n')
							writer.flush()
					else:
						pass
	def calculate_proportion(self, layer_name):
		#self.calculate_all_w_star()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'w_star' in param_state and param_state['name'] == layer_name:
					#print(layer_name)
					self.calculate_w_star_by_layer(layer_name)
					if len(p.data.size())==4:
						ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
						ts_norm = torch.norm(ts_reshape, 2, 1)
						num_selected_filters = torch.sum(ts_norm != 0).item()
						return num_selected_filters/p.data.size()[0]
					elif len(p.data.size())==2:
						num_selected_units = (param_state['w_star'] > 0.0).sum().item()
						return num_selected_units/p.data.size()[0] * p.data.size()[1]
					else:
						pass



	def calculate_norm(self, layer_name):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if param_state['name'] == layer_name:
					layer_norm = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1)
		return layer_norm.cpu().detach().numpy()


	def cal_prune_thre(self, percent, layer_name):
		# prune layer according to given percent and layer name
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				if param_state['name'] in layer_name and 'prune_order' in param_state:
					print(param_state['name'])
					order = param_state['prune_order'].cpu().detach().numpy()
					threshold = np.percentile(order, percent)
					print('Threshold : ', threshold)
		return threshold


	def update_prune_order(self, epoch):
		self.calculate_all_w_star()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'z_buffer' in param_state:
					if len(p.data.size())==4:
						if 'epoch_record' not in param_state:
							param_state['epoch_record'] = torch.zeros_like(p.data).add_(2000.0)
							mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
							param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
							epoch_per_filer, _ = torch.min(torch.reshape(param_state['epoch_record'], (param_state['epoch_record'].shape[0], -1)), dim=1) 
							param_state['prune_order'] = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1) - epoch_per_filer
						else:
							mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
							#print(mask.size())
							#print(param_state['epoch_record'].size())
							param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
							epoch_per_filer, _ = torch.min(torch.reshape(param_state['epoch_record'], (param_state['epoch_record'].shape[0], -1)), dim=1) 
							param_state['prune_order'] = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1) - epoch_per_filer		
					elif len(p.data.size()) == 2:
						if 'epoch_record' not in param_state:
							param_state['epoch_record'] = torch.zeros_like(p.data).add_(2000.0)
							mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
							param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
							param_state['prune_order'] = torch.abs(p.data) - param_state['epoch_record']
						else:
							mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
							param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),  param_state['epoch_record'])
							#param_state['prune_order'] = torch.abs(param_state['w_star']) - param_state['epoch_record']
							param_state['prune_order'] = torch.abs(p.data) - param_state['epoch_record']
					else:
						pass


	def prune_layer_by_order_by_name(self, percent, layer_name, prune_bias):
		# prune layer according to given percent and layer name
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				if param_state['name']==layer_name and 'prune_order' in param_state:
					print(param_state['name'])
					order = param_state['prune_order'].cpu().detach().numpy()
					threshold = np.percentile(order, percent)
					if len(p.data.size())==4:
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data[threshold > param_state['prune_order'], :, :, :] = 0.0
						if prune_bias:
							for k in range(iter + 1, len(group['params'])):
								p_n = group['params'][k]
								param_state_n = self.state[p_n]
								if param_state_n['name'] == layer_name.replace('weight', 'bias'):
									print(param_state_n['name'])
									param_state_n['original_params'] = copy.deepcopy(p_n.data)
									p_n.data[threshold > param_state['prune_order']] = 0.0
					elif len(p.data.size())==2:
						num_selected_units = (param_state['w_star'] > 0.0).sum().item()
						mask = (torch.gt(param_state['prune_order'], threshold)).float()
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data = p.data * mask
					else:
						pass
				elif param_state['name'] in layer_name and 'prune_order' not in param_state:
					print('Please Update Order First')
				else:
					pass


	def prune_layer_by_order_by_list(self, percent, layer_name, prune_bias):
		# prune layer according to given percent and layer name
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				if param_state['name'] in layer_name and 'prune_order' in param_state:
					print(param_state['name'])
					order = param_state['prune_order'].cpu().detach().numpy()
					threshold = np.percentile(order, percent)
					if len(p.data.size())==4:
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data[threshold > param_state['prune_order'], :, :, :] = 0.0
						if prune_bias:
							for k in range(iter + 1, len(group['params'])):
								p_n = group['params'][k]
								param_state_n = self.state[p_n]
								if param_state_n['name'] ==  param_state['name'].replace('weight', 'bias'):
									print(param_state_n['name'])
									param_state_n['original_params'] = copy.deepcopy(p_n.data)
									p_n.data[threshold > param_state['prune_order']] = 0.0
					elif len(p.data.size())==2:
						num_selected_units = (param_state['w_star'] > 0.0).sum().item()
						mask = (torch.gt(param_state['prune_order'], threshold)).float()
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data = p.data * mask
					else:
						pass
				elif param_state['name'] in layer_name and 'prune_order' not in param_state:
					print('Please Update Order First')
				else:
					pass


	def recover(self):
		#in use_w_star or prune_layer, params are changed. so using recover() can give params original value
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'original_params' in param_state:
					p.data = param_state['original_params']


	def extract_layer_weights(self, layer_name,  number_select):
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				print(param_state['name'])
				if len(p.data.size())==4 and 'prune_order' in param_state and param_state['name'] == layer_name:
					sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
					selected_filters = p.data[indices[0 : number_select], :, :, :]
					return selected_filters
				elif len(p.data.size())==2 and 'prune_order' in param_state and param_state['name'] == layer_name:
					sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
					selected_filters = p.data[:, indices[0 : number_select]]
					return selected_filters 
				else:
					pass

	def extract_conv_and_fc_weights(self, layer_name, number_select):
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				print(param_state['name'])
				if len(p.data.size())==4 and 'prune_order' in param_state and param_state['name'] == layer_name:
					sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
					selected_filters = p.data[indices[0 : number_select], :, :, :]
					for k in range(iter + 1, len(group['params'])):
						p_fc = group['params'][k]
						if len(p_fc.data.size()) == 2:
							break
					step = int(p_fc.data.size()[1] / p.data.size()[0])
					#print(p_fc.data.size()[1])
					#print(p.data.size())
					fc_indice = []
					for j in range(len(indices[0 : number_select])):
						fc_indice.extend(range(indices[j]*step, (indices[j]+1)*step))
					#print(fc_indice)
					#print(p_fc.data.size())
					selected_weights = p_fc.data[:, fc_indice]
					return selected_filters, selected_weights
				else:
					pass
	def ortho_init(self, init_matrix, coordinate_matrix):
		_, w, h, cin =  coordinate_matrix.size()
		coordinate_matrix_v = coordinate_matrix.view(-1, w*h*cin).transpose(0, 1)
		print(coordinate_matrix_v.size())
		q, r = torch.qr(coordinate_matrix_v)
		r_d = torch.diag(r)
		print(r_d)
		sorted_d, indices = torch.sort(torch.abs(r_d), descending=True)
		eps = 1e-10
		rnk = min(coordinate_matrix_v.size()) - torch.ge(sorted_d, eps).sum()
		print('rank :' ,rnk)
		basis = q[:, indices[rnk :]].transpose(0, 1)
		basis = basis.view(-1, w, h, cin)
		stdv = 1. / math.sqrt(w*h*cin)
		nll_count = 0
		for k in range(init_matrix.size()[0]):
			if nll_count < basis.size()[0]:
				init_matrix[k, :, :, :] = torch.clamp(basis[k, :, :, :],-stdv, stdv)
				nll_count += 1
			else:
				init_matrix[k].uniform_(-stdv, stdv)


	def reinitialize(self, layer_list, percent, reinitialize_threshold):
		print('Reinitialize')
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				if 'prune_order' in param_state and param_state['name'] in layer_list and len(p.data.size()) == 4:
					proportion = self.calculate_proportion(param_state['name'])
					print(param_state['name'])
					print('Proportion : ', proportion)
					if proportion > reinitialize_threshold:
						sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
						border = int((1-percent/100) * p.data.size()[0]) 
						keep_indice = indices[range(0, border)]
						reinit_indice = indices[range(border, p.data.size()[0])]
						self.ortho_init(p.data[reinit_indice, :, :, :], p.data[keep_indice, :, :, :])
						param_state['prune_order'][reinit_indice] = -2000.0
						param_state['gamma_buffer'][reinit_indice] = 0.0
						param_state['z_buffer'][reinit_indice] = 0.0
						param_state['w_star'][reinit_indice] = 0.0
						for k in range(iter + 1, len(group['params'])):
							p_b = group['params'][k]
							p_b_state = self.state[p_b]
							if len(p_b.data.size()) == 1 and p_b_state['name'] == param_state['name'].replace('bias', 'weight'):
								p_b.data[reinit_indice] = 0.0


	
	def calculate_mask(self, layer_name):
		self.calculate_all_w_star()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'w_star' in param_state and param_state['name'] == layer_name:
					if len(p.data.size())==4:
						ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
						ts_norm = torch.norm(ts_reshape, 2, 1)
						num_selected_filters = torch.sum(ts_norm != 0).item()
						mask = torch.ones_like(param_state['w_star'])
						mask[ts_norm != 0, :, :, :] = 0
						return mask
					elif len(p.data.size())==2:
						return torch.ones_like(param_state['w_star'])
						#return torch.gt(torch.abs(param_state['w_star']), 0.0).float()
					else:
						pass

	def step_with_freeze(self, freeze=True):
		loss = None
		for group in self.param_groups:
			mu = group['mu']
			kappa = group['kappa']
			lr_kappa = group['lr'] * group['kappa']
			lr_gamma = group['lr'] / mu
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data
				param_state = self.state[p]
				if momentum != 0:
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
						buf.mul_(momentum).add_(d_p)
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					d_p = buf
				if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
					d_p.add_(weight_decay, p.data)
				if  'z_buffer' in param_state:
					if freeze:
						mask = self.calculate_mask(param_state['name'])
					else:
						mask = torch.ones_like(d_p)
					new_grad = d_p * lr_kappa + (p.data - param_state['gamma_buffer']) * lr_kappa / mu 
					new_grad = new_grad * mask
					last_p = copy.deepcopy(p.data)				
					p.data.add_(-new_grad)
					param_state['z_buffer'].add_(-lr_gamma, mask *(param_state['gamma_buffer'] - last_p))
					if len(p.data.size()) == 2:
						param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], 1)
					elif len(p.data.size()) == 4:
						param_state['gamma_buffer'] = kappa * self.shrink_group(param_state['z_buffer'])
					else:
						pass
				else:
					p.data.add_(-lr_kappa, d_p)#for bias update as vanilla sgd



	

	def print_network(self):
		print('Printing Network')
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				print(param_state['name'], p.data.size())

	def generate_dict(self):
		net_dict = {}
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				s_name = param_state['name'].replace('module.',  '')
				if s_name == 'conv1.weight':
					net_dict['conv1.out'] = p.data.size()[0]
				elif s_name == 'fc.weight':
					net_dict['fc.in'] = p.data.size()[1]
				elif len(p.data.size()) == 4:
					n_name = param_state['name'].replace('module.',  '')
					n_name = n_name.replace('.weight', '')
					print(n_name)
					net_dict[n_name + '.in'] = p.data.size()[1]
					net_dict[n_name + '.out'] = p.data.size()[0]
				else:
					pass
		return net_dict
	def get_size(self, layer_name):
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				if param_state['name'] == layer_name:
					return p.data.size()
	def get_z_state_dict(self):
		z_dict = OrderedDict()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if  'z_buffer' in param_state:
					z_dict[param_state['name']] = param_state['z_buffer'].cpu().detach()
		return z_dict

	def get_mask(self):
		N = 0
		N_s = 0
		mask = OrderedDict()
		torch.cuda.empty_cache()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				N += p.size().numel()
				if  'z_buffer' in param_state:
					if len(p.data.size()) == 4 or len(p.data.size()) == 2:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
						num_selected_units = (torch.abs(param_state['w_star']) > 0.0).cpu().sum().item()
						N_s += num_selected_units
						mask[param_state['name']] = torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)
				else:
					N_s += p.size().numel()
		print('Ratio :' , float(N_s) / float(N))
		return mask, float(N_s) / float(N)

	def check_sparsity(self, recorder=None):
		N = 0
		N_s = 0
		torch.cuda.empty_cache()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				N += p.size().numel()
				N_s += torch.gt(torch.abs(p.data), 0.0).float().sum()
		print('Sparsity :' , float(N_s) / float(N))
		if recorder:
			recorder.write('Checking Sparsity :')
			recorder.write(str(float(N_s) / float(N)))
			recorder.write('\n')
			recorder.flush()
		return float(N_s) / float(N)
