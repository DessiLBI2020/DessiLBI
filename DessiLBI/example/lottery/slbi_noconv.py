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

    def __init__ (self, params, lr=required, kappa=1, mu=100, weight_decay=0, momentum=0.9, dampening=0, nesterov=False, fc_lambda=0.1, conv_lambda=0.1, bn_lambda=0.1):
        defaults = dict(lr=lr, kappa=kappa, mu=mu, weight_decay=weight_decay, momentum=momentum, dampening=dampening, nesterov=nesterov, fc_lambda=fc_lambda, conv_lambda=conv_lambda, bn_lambda=bn_lambda)
        print('*******************************************')
        for key in defaults:
            print(key, ' : ', defaults[key])
        print('*******************************************')
        super(SLBI, self).__init__(params, defaults)


    def get_gamma_state_dict(self):
        gamma_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'gamma_buffer' in param_state:
                    gamma_dict[param_state['name']] = param_state['gamma_buffer']
        return gamma_dict


    def get_z_state_dict(self):
        z_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state:
                    z_dict[param_state['name']] = param_state['z_buffer']
        return z_dict


    def save_gamma_state_dict(self, path):
        gamma_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'gamma_buffer' in param_state:
                    gamma_dict[param_state['name']] = param_state['gamma_buffer']
        torch.save(gamma_dict,path)

    def generate_strong_mask_dict(self):
        mask = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state and len(p.data.size()) == 4:
                    mask[param_state['name']] = torch.gt(torch.abs(param_state['gamma_buffer']), 0.0).float()
        return mask

    def generate_weak_mask_dict(self):
        mask = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state:
                    mask[param_state['name']] = 1 - torch.gt(torch.abs(param_state['gamma_buffer']), 0.0).float()
        return mask




    def load_mask(self, mask):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['name'] in mask.keys():
                    state['mask'] = mask[state['name']]

    def apply_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'mask' in state:
                    p.data = p.data * state['mask']


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
                    if param_state['name'] in layer_list and len(p.data.size()) != 1:
                        param_state['z_buffer'] = torch.zeros_like(p.data)
                        param_state['gamma_buffer'] = torch.zeros_like(p.data)



    def step(self, closure=None, record=False, path='./'):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            mu = group['mu']
            fc_lambda = group['fc_lambda']
            conv_lambda = group['conv_lambda']
            bn_lambda = group['bn_lambda']
            kappa = group['kappa']
            lr_kappa = group['lr'] * group['kappa']
            lr_gamma = group['lr'] / mu
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and 'bias' not in param_state['name']:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if  'z_buffer' in param_state:
                    new_grad = d_p  + (p.data - param_state['gamma_buffer']) / mu
                    last_p = copy.deepcopy(p.data)
                    d_p = new_grad
                    param_state['z_buffer'].add_(-lr_gamma, param_state['gamma_buffer'] - last_p)
                    if len(p.data.size()) == 1:
                        pass
                    elif len(p.data.size()) == 2:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], fc_lambda)
                    elif len(p.data.size()) == 4:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'],fc_lambda)
                    else:
                        pass
                p.data.add_(-lr_kappa, d_p)


    def get_numel(self, x):
        n = 1
        for i in range(len(x)):
            n *= x[i]
        return n

    def calculate_all_w_star(self, use_sparse=True, file_obj=None):
        N = 0
        N_s = 0
        torch.cuda.empty_cache()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                N += p.size().numel()
                if  'z_buffer' in param_state:
                    if len(p.data.size()) == 1:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                        num_selected_units = (torch.abs(param_state['w_star']) > 0.0).cpu().sum().item()
                        N_s += num_selected_units
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']

                        file_obj.write('###############################################')
                        file_obj.write('\n')
                        file_obj.write('max: ')
                        file_obj.write(str(torch.max(param_state['w_star']).item()))
                        file_obj.write('\n')
                        file_obj.write('min: ')
                        file_obj.write(str(torch.min(param_state['w_star']).item()))
                        file_obj.write('\n')
                        file_obj.write(' Sparsity of ' + param_state['name'])
                        file_obj.write('\n')
                        file_obj.write('number of all channels:')
                        file_obj.write(str(p.data.size()[0]))
                        file_obj.write('\n')
                        file_obj.write('number of selected channels:')
                        file_obj.write(str(num_selected_units))
                        file_obj.write('\n')
                        file_obj.write('ratio of selected channels:')
                        file_obj.write(str(num_selected_units/(p.data.size()[0])))
                        file_obj.write('\n')


                    elif len(p.data.size()) == 2:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                        num_selected_units = (torch.abs(param_state['w_star']) > 0.0).cpu().sum().item()
                        N_s += num_selected_units
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']



                        file_obj.write('###############################################')
                        file_obj.write('\n')
                        file_obj.write('max: ')
                        file_obj.write(str(torch.max(param_state['w_star']).item()))
                        file_obj.write('\n')
                        file_obj.write('min: ')
                        file_obj.write(str(torch.min(param_state['w_star']).item()))
                        file_obj.write('\n')
                        file_obj.write(' Sparsity of ' + param_state['name'])
                        file_obj.write('\n')
                        file_obj.write('number of all channels:')
                        file_obj.write(str(p.data.size()[0] * p.data.size()[1] ))
                        file_obj.write('\n')
                        file_obj.write('number of selected channels:')
                        file_obj.write(str(num_selected_units))
                        file_obj.write('\n')
                        file_obj.write('ratio of selected channels:')
                        file_obj.write(str(num_selected_units/(p.data.size()[0]*p.data.size()[1])))
                        file_obj.write('\n')

                    elif len(p.data.size()) == 4:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                        num_selected_units = (torch.abs(param_state['w_star']) > 0.0).cpu().sum().item()
                        N_s += num_selected_units
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']


                        file_obj.write('###############################################')
                        file_obj.write('\n')
                        file_obj.write('max: ')
                        file_obj.write(str(torch.max(param_state['w_star']).item()))
                        file_obj.write('\n')
                        file_obj.write('min: ')
                        file_obj.write(str(torch.min(param_state['w_star']).item()))
                        file_obj.write('\n')
                        file_obj.write(' Sparsity of ' + param_state['name'])
                        file_obj.write('\n')
                        file_obj.write('number of channels:')
                        file_obj.write(str(p.data.size()[0] ))
                        file_obj.write('\n')

                        file_obj.write('number of all channels:')
                        file_obj.write(str(p.data.size()[0] * p.data.size()[1] * p.data.size()[2] * p.data.size()[3]))
                        file_obj.write('\n')
                        file_obj.write('number of selected channels:')
                        file_obj.write(str(num_selected_units))
                        file_obj.write('\n')
                        file_obj.write('ratio of selected channels:')
                        file_obj.write(str(num_selected_units/(p.data.size()[0] * p.data.size()[1] * p.data.size()[2] * p.data.size()[3])))
                        file_obj.write('\n')
                else:
                    N_s += p.size().numel()
        file_obj.write('###############################################')
        file_obj.write('\n')
        file_obj.write('Model Sparsity')
        file_obj.write(str(N_s/N))
        file_obj.write('\n')
        file_obj.write('Model Param')
        file_obj.write(str(N))
        file_obj.write('\n')
        file_obj.write('Sparsity Param')
        file_obj.write(str(N_s))
        file_obj.write('\n')


    def shrink(self, s_t, lam):
		#proximal mapping for 2-d weight(fc layer) and bn weight
        gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
        return gamma_t


    def recover(self):
        #in use_w_star or prune_layer, params are changed. so using recover() can give params original value
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'original_params' in param_state:
                    p.data = param_state['original_params']
                    del param_state['original_params']
        torch.cuda.empty_cache()


