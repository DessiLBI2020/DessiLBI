"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np

import torch
from torch.optim import Adam
# from torchvision import models
from torch import nn

from misc_functions import preprocess_image, recreate_image

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, (1,1,32, 32)))
        # Create the folder to export images if not exists
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.model.convnet.relu5.register_forward_hook(hook_function)
        # self.model.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image,resize_im=False)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=10, weight_decay=1e-8,amsgrad=True)
        Tvloss=TVLoss(tv_loss_lambda)
        for i in range(1, 151):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            # for index, layer in enumerate(self.model):
            #     # Forward pass layer by layer
            #     # x is not used after this point because it is only needed to trigger
            #     # the forward hook function
            #     x = layer(x)
            #     # Only need to forward until the selected layer is reached
            #     if index == self.selected_layer:
            #         # (forward hook function triggered)
            #         break
            # tv_loss = Tvloss(x)
            x = self.model(x)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(float(loss.data.numpy())))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            if not os.path.exists('./generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter)):
                os.makedirs('./generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter))
            # if i % 5 == 0:
            #     cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
            #                 '_f' + str(self.selected_filter) + '/iter'+str(i)+'.jpg',
            #                 self.created_image)
            if i % 150 == 0:
                cv2.imwrite('./generated/' + str(self.selected_layer) +
                            '_' + str(self.selected_filter) + 'iter'+str(i)+'_filter_'+str(index)+'_norm{:.4f}'.format(linear[index])+'.jpg',
                            self.created_image)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 91):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            if i % 5 == 0:
                cv2.imwrite('./generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
                            self.created_image)


if __name__ == '__main__':
    from get_small_model import *
    for index in range(120):
        cnn_layer = "c5"#useless
        filter_pos =index
        tv_loss_lambda=0.00001
        # Fully connected layer is not needed
        # pretrained_model = models.resnet18(pretrained=True)
        pretrained_model = get_lenet()
        print(type(pretrained_model))
        print(pretrained_model)
        linear=pretrained_model.convnet.c5.weight.data
        print(linear.shape)
        linear=torch.reshape(linear,[120,-1])
        linear=torch.sum(linear*linear,dim=1)
        print(linear)
        for i in range(len(linear)):
            # if linear[i]>0.0 and linear[i]<1:
            #     continue
            print("filter index",i,"\t","filter norm{:.4f}".format(linear[i]))
        print(index)
        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

        # Layer visualization with pytorch hooks
        layer_vis.visualise_layer_with_hooks()

        # Layer visualization without pytorch hooks
        # layer_vis.visualise_layer_without_hooks()
