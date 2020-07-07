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
import numpy as np 
from slbi_opt import SLBI
import copy
import math

class SLBI_ToolBox(SLBI):

	def use_w_star(self):
		#use sparse params to replace original params
		self.calculate_all_w_star()
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'w_star' in param_state:
					if len(p.data.size())==4:
						ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
						ts_norm = torch.norm(ts_reshape, 2, 1)
						num_selected_filters = torch.sum(ts_norm != 0).item()
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data = param_state['w_star']
						print('max:', torch.max(param_state['w_star']))
						print('min:', torch.min(param_state['w_star']))
						print('number of filters: ',  p.data.size()[0])
						print('number of selected filter:', num_selected_filters )
					elif len(p.data.size())==2:
						num_selected_units = (param_state['w_star'] > 0.0).sum().item()
						param_state['original_params'] = copy.deepcopy(p.data)
						p.data = param_state['w_star']
						print('max:', torch.max(param_state['w_star']))
						print('min:', torch.min(param_state['w_star']))
						print('number of filters: ',  p.data.size()[0] * p.data.size()[1])
						print('number of selected units:', num_selected_units )
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







