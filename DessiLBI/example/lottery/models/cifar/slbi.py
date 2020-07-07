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

    def __init__ (self, params, lr=required, kappa=1, mu=100, weight_decay=0, momentum=0.9, dampening=0, nesterov=False, fc_lambda=1, conv_lambda=1, bn_lambda=1):
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
                if  'z_buffer' in param_state:
                    gamma_dict[param_state['name']] = param_state['z_buffer']
        return gamma_dict


    def save_gamma_state_dict(self, path):
        gamma_dict = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state:
                    gamma_dict[param_state['name']] = param_state['z_buffer']
        torch.save(gamma_dict,path)

    def generate_strong_mask_dict(self):
        mask = OrderedDict()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state and len(p.data.size()) == 4:                     
                    mask[param_state['name']] = torch.gt(torch.abs(param_state['gamma_buffer']), 0.0).float()
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
                    if param_state['name'] in layer_list:
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
            #bn_lambda = group['bn_lambda']
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
                    d_p.add_(weight_decay, param_state['z_buffer'])
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
                    # how to update z ?
                    param_state['z_buffer'].add_(-lr_gamma, param_state['gamma_buffer'] - last_p)
                 #   if len(p.data.size()) == 1 and 'bias' not in param_state['name']:
                  #      param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], fc_lambda)
                    if len(p.data.size()) == 2:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], fc_lambda)
                    elif len(p.data.size()) == 4:# use conv lambda again
                        param_state['gamma_buffer'] = kappa * self.shrink_conv(param_state['z_buffer'], conv_lambda)
                    else:
                        pass
                # how to update W?
                p.data.add_(-lr_kappa, d_p)
                if "gamma_buffer" in param_state and 'z_buffer' in param_state:
                    sign=((p.data-param_state["gamma_buffer"]).sign()+(last_p-param_state["gamma_buffer"]).sign())==0
                    p.data[sign]=param_state["gamma_buffer"][sign]


    def step_with_mask(self, closure=None, record=False, path='./'):
        loss = None
        if closure is not None:
            loss = closure()        
        for group in self.param_groups:
            mu = group['mu']
            fc_lambda = group['fc_lambda']
            #conv_lambda = group['conv_lambda']
            #bn_lambda = group['bn_lambda']
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


                if weight_decay != 0 and 'bias' not in param_state['name']:
                    d_p.add_(weight_decay, param_state['gamma_buffer'])
                if  'z_buffer' in param_state:
                    new_grad = d_p  + (p.data - param_state['gamma_buffer']) / mu 
                    last_p = copy.deepcopy(p.data)              
                    d_p = new_grad 
                    param_state['z_buffer'].add_(-lr_gamma, param_state['gamma_buffer'] - last_p)
                    #if len(p.data.size()) == 1 and 'bias' not in param_state['name']:
                     #   param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], fc_lambda)
                    if len(p.data.size()) == 2:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], fc_lambda)
                    elif len(p.data.size()) == 4:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], conv_lambda)
                    else:
                        pass
                if 'mask' in param_state:
                    d_p = d_p * param_state['mask']
                p.data.add_(-lr_kappa, d_p)


    def calculate_all_w_star(self, use_sparse=True,writer=None,epoch=0):
        total_param=0
        selected_parm=0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if  'z_buffer' in param_state:
                    #if len(p.data.size()) == 1:                      
                     #   param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                     #   num_selected_units = (torch.abs(param_state['w_star']) > 0.0).sum().item()
                     #   if use_sparse:
                      #      param_state['original_params'] = copy.deepcopy(p.data)
                      #      p.data = param_state['w_star']
                        # print('###############################################')
                        # print('max:', torch.max(param_state['w_star']))
                        # print('min:', torch.min(param_state['w_star']))
                      #  print(' Sparsity of ' + param_state['name'])
                        # print('number of selected channels:', num_selected_units)
                        # print('number of all channels:',  p.data.size()[0])
                       # print('ratio of selected channels:', num_selected_units/(p.data.size()[0]))
                    if len(p.data.size()) == 2:                      
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                        num_selected_units = (torch.abs(param_state['w_star']) > 0.0).sum().item()
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']
                    elif len(p.data.size()) == 4: #modify
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                        num_selected_units = (torch.abs(param_state['w_star']) > 0.0).sum().item()
                        if use_sparse:
                            param_state['original_params'] = copy.deepcopy(p.data)
                            p.data = param_state['w_star']
                    if writer:
                        writer.add_scalar("sparity of "+param_state['name'],num_selected_units/p.data.numel(),epoch)
                        total_param+=p.data.numel()
                        selected_parm+=num_selected_units

        if writer:
            print('Total Sparsity : ', selected_parm/total_param,epoch)
            writer.add_scalar("total sparity",selected_parm/total_param,epoch)


    def shrink(self, s_t, lam):
		#proximal mapping for 2-d weight(fc layer) and bn weight
        gamma_t = s_t.sign() * (torch.max(s_t.abs() - lam * torch.ones_like(s_t), torch.zeros_like(s_t)))
        return gamma_t


    def shrink_conv(self, s_t, lam):
        #proximal mapping for 4-d weight
        ts_reshape = torch.reshape(s_t, (s_t.shape[0],-1))
        ts_norm = torch.norm(ts_reshape,2,1)
        ts_norm = torch.unsqueeze(ts_norm, 1)
        ts_norm = torch.unsqueeze(ts_norm, 2)
        ts_norm = torch.unsqueeze(ts_norm, 3)
        ts_norm = ts_norm.repeat(1, s_t.size()[1],s_t.size()[2], s_t.size()[3] )
        gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * ts_norm), torch.zeros_like(s_t)))
        return gamma_t

    def recover(self):
        #in use_w_star or prune_layer, params are changed. so using recover() can give params original value
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'original_params' in param_state:
                    p.data = param_state['original_params']


    def shrink_group(self, ts):
		# shrinkage for 4-d weight(conv layer)
        ts_reshape = torch.reshape(ts,(ts.shape[0],-1))
        ts_norm = torch.norm(ts_reshape,2,1)
        ts_shrink = torch.max(torch.zeros_like(ts_norm),torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm),ts_norm))
        ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape,0,1),ts_shrink),0,1)
        ts_return = torch.reshape(ts_return,ts.shape)
        return ts_return

    def record_param_hist(self,writer,epoch):
        for group in self.param_groups:
            for p in group['params']:
                # print(self.state[p])
                if "z_buffer" in self.state[p].keys():
                    writer.add_histogram(self.state[p]["name"]+"Z",self.state[p]["z_buffer"],epoch)
                # param_state = self.state[p]
                # if 'original_params' in param_state:
                #     p.data = param_state['original_params']


