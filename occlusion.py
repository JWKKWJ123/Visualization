from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from tqdm import tqdm_notebook
import multiprocessing
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import cv2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#function for computing occlusion maps for 3D brain images
def occlusion(model, image_tensor, target_class=None, size=30, stride=25, occlusion_value=0, apply_softmax=True, three_d=None, resize=True, cuda=False, verbose=False):
    """
    Perform occlusion (Zeiler & Fergus 2014) to determine the relevance of each image pixel 
    for the classification decision. Return a relevance heatmap over the input image.
    
    Note: The current implementation can only handle 2D and 3D images. 
    It usually infers the correct image dimensions, otherwise they can be set via the `three_d` parameter. 
    
    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode. 
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap. 
                      If `None` (default), use the most likely class from the `model`s output.
        size (int): The size of the occlusion patch.
        stride (int): The stride with which to move the occlusion patch across the image.
        occlusion_value (int): The value of the occlusion patch.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained 
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        three_d (boolean): Whether the image is 3 dimensional (e.g. MRI scans). 
                           If `None` (default), infer from the shape of `image_tensor`. 
        resize (boolean): The output from the occlusion method is usually smaller than the original `image_tensor`. 
                          If `True` (default), the output will be resized to fit the original shape (without interpolation). 
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.
        
    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel. 
    """
    
    # TODO: Try to make this better, i.e. generalize the method to any kind of input.
    if three_d is None:
        three_d = (len(image_tensor.shape) == 4)  # guess if input image is 3D
    if cuda:
        image_tensor = image_tensor.to(device)
    output = model(Variable(image_tensor[None], requires_grad=False))
    if apply_softmax:
        output = F.softmax(output)
 
    output_class = output.max(1)[1].data.cpu().numpy()[0]
    if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
    if target_class is None:
        target_class = output_class
        print(target_class)
    unoccluded_prob = output.data[0, target_class]
    print('unoccluded_prob:',unoccluded_prob)
        
    width = image_tensor.shape[1]
    height = image_tensor.shape[2]
    
    xs = range(0, width, stride)
    ys = range(0, height, stride)
    
    # TODO: Maybe use torch tensor here.
    if three_d:
        depth = image_tensor.shape[3]
        zs = range(0, depth, stride)
        relevance_map = np.zeros((len(xs), len(ys), len(zs)))
    else:
        relevance_map = np.zeros((len(xs), len(ys)))
    
    if verbose:
        xs = tqdm_notebook(xs, desc='x')
        ys = tqdm_notebook(ys, desc='y', leave=False)
        if three_d:
            zs = tqdm_notebook(zs, desc='z', leave=False)
            
    image_tensor_occluded = image_tensor.clone()  # TODO: Check how long this takes.
    
    if cuda:
        image_tensor_occluded = image_tensor_occluded.to(device)
    
    for i_x, x in enumerate(xs):
        x_from = max(x - int(size/2), 0)
        x_to = min(x + int(size/2), width)
        
        for i_y, y in enumerate(ys):
            y_from = max(y - int(size/2), 0)
            y_to = min(y + int(size/2), height)
            
            if three_d:
                for i_z, z in enumerate(zs):
                    
                    z_from = max(z - int(size/2), 0)
                    z_to = min(z + int(size/2), depth)

                    #if verbose: print('Occluding from x={} to x={} and y={} to y={} and z={} to z={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))

                    image_tensor_occluded.copy_(image_tensor)
                    image_tensor_occluded[:, x_from:x_to, y_from:y_to, z_from:z_to] = occlusion_value
            
                    # TODO: Maybe run this batched.
                    outputs = model(Variable(image_tensor_occluded[None], requires_grad=False))
                    
                    if apply_softmax:
                        output = F.softmax(output)
                    
                    occluded_prob = outputs.data[0, target_class]
                    
                    if torch.where(output[0][0] > 0.5, 1, 0).cpu().detach().numpy() == 1:
                       relevance_map[i_x, i_y, i_z] =  unoccluded_prob - occluded_prob
                    if torch.where(output[0][0] > 0.5, 1, 0).cpu().detach().numpy() == 0:
                       relevance_map[i_x, i_y, i_z] =  occluded_prob - unoccluded_prob
                       
                    
            else:
                #if verbose: print('Occluding from x={} to x={} and y={} to y={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))
                image_tensor_occluded.copy_(image_tensor)
                image_tensor_occluded[:, x_from:x_to, y_from:y_to] = occlusion_value
                
                # TODO: Maybe run this batched.
                output = model(Variable(image_tensor_occluded[None], requires_grad=False))
                if apply_softmax:
                    output = F.softmax(output)
                
                occluded_prob = output.data[0, target_class]
                relevance_map[i_x, i_y] = unoccluded_prob - ocluded_prob
                
    relevance_map = np.maximum(relevance_map, 0)
                    

                
    return relevance_map
