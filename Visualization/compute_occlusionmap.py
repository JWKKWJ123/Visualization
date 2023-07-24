from Glo-CNN import Global
import math
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import einops
import glob
import time
from sklearn.metrics import accuracy_score,recall_score,roc_curve, classification_report,confusion_matrix,precision_score,roc_auc_score, auc
import random
from occlusion import occlusion
from scipy import ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



#function for demonstrating 3D heatmap in 2D slices 
def plot_slices(struct_arr, num_slices=7, cmap='gray', vmin=None, vmax=None, overlay=None, overlay_cmap=mycmap, overlay_vmin=None, overlay_vmax=None, _class=None,iteration=0):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis
    
    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colomap for the overlay (default: `alpha_to_red_cmap`). 
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    
    overlay[overlay<np.percentile(overlay, 70)]=0
    
    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax) 
        
    fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ['x', 'y', 'z']):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            plt.sca(ax)
            plt.axis('off')
            plt.imshow(ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90), vmin=vmin, vmax=vmax, 
                       cmap=cmap, interpolation=None)
            plt.text(0.03, 0.97, '{}={}'.format(axis_label, i_slice), color='white', 
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            
            if overlay is not None:
                plt.imshow(ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90), cmap=overlay_cmap, 
                           vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None,alpha=0.5)
                #save the group-level heatmap of different class and different testing set
				if _class == 0:
                   plt.savefig('_{}.jpg'.format(iteration))
                if _class == 1:
                   plt.savefig('_{}.jpg'.format(iteration))

				   
#dataloader
#only output the occlution map of subjects in the split training set, because the occlution map is used in feature selection
class SelfDataset(Dataset):
    def __init__(self, data_dir,ki=0, K=5, typ='train'):
        self.images = []
        self.labels = []
        self.names = []
        self.subnames = []

        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images)) 
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
			#subname is the subject ID
            self.subnames += [os.path.relpath(imgs, data_dir)[2:-4] for imgs in images]
        #set split training set (90%) and testing set (10%)    
        ss1=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state= 0)
        train_index, test_index = ss1.split(self.images, self.labels)
        test_index = train_index[1]
        train_index = train_index[0]
        self.images, X_test = np.array(self.images)[train_index], np.array(self.images)[test_index]#???????
        self.labels, y_test = np.array(self.labels)[train_index], np.array(self.labels)[test_index]#???????
        self.subnames, name_test = np.array(self.subnames)[train_index], np.array(self.subnames)[test_index]#???????
        
        #set K fold cross-validation in the split training set
        sfolder = StratifiedKFold(n_splits=K,random_state=0,shuffle=True)
        i=0
		#set training and validation set in each fold
        for train, test in sfolder.split(self.images,self.labels):
            if i==ki:
               if typ == 'val':
                  self.crossimages = np.array(self.images)[test]
                  self.crosslabels = np.array(self.labels)[test]
                  self.crossnames = np.array(self.subnames)[test]
               elif typ == 'train':
                  self.crossimages = np.array(self.images)[train]
                  self.crosslabels = np.array(self.labels)[train]
                  self.crossnames = np.array(self.subnames)[train]  
            i=i+1
           
    def __getitem__(self, idx):
        image = np.load(self.crossimages[idx])
        image = image[15:165,20:200,15:165]
        image = (image - np.mean(image))/ np.std(image)
        label = self.crosslabels[idx]
        name = self.crossnames[idx]
        return image, label, name
    def __getname__(self):
        return self.crossnames
    def __len__(self):
        return len(self.crossimages)


#set the gradient color used in heatmap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = ['black','blue','orange','gold','tomato','red']
color_list = ListedColormap(colors)
mycmap = LinearSegmentedColormap.from_list('mycmap', colors)

root = '  '
#load the brain template used as the background of heatmap 
x_train = np.load("/templete.npy")
x_train = x_train[15:165,20:200,15:165]
#normalization
x_train = ((x_train - np.mean(x_train))/ (np.max(x_train)-np.min(x_train))+1)/2

#set hyperparameters of occlusion method, occlusion stride and size of occlusion patch are the most important hyperparameter
occ_stride = 5
occ_size = 20
K= 10

occ_all = np.zeros([1,150//occ_stride,180//occ_stride,150//occ_stride])
for ki in range(K):
    occ_AD = np.zeros([1,150//occ_stride,180//occ_stride,150//occ_stride])
    occ_CN = np.zeros([1,150//occ_stride,180//occ_stride,150//occ_stride])
    testset = SelfDataset(root, ki=ki, K=K, typ='val')
    test_data_size = len(testset)
    test_dataloader = DataLoader(
         dataset=testset,
         batch_size=1,shuffle = True
     )
    logs = []
	#load models trained in cross-validation 
    model = torch.load("_{}.pth".format(ki),map_location=torch.device(device) )
    model.eval()
    i=0
	#compute subject-level occlusion map in testing set 
    for data in test_dataloader:

            imgs, targets ,biomarker = data
            #set occlusion value to the grayscale value of the background area of the image after normalization
            background_value = imgs[0,0,0,0]
            imgs = imgs.to(torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            targets = targets.to(device)
            relevance_map_occlusion = occlusion(model, imgs, size=occ_size, stride=occ_stride, occlusion_value=background_value,cuda=True, verbose=True,three_d=True,apply_softmax=False)
            print('occ:',np.max(relevance_map_occlusion),np.mean(relevance_map_occlusion),np.min(relevance_map_occlusion))
            occlusion_map = np.expand_dims(relevance_map_occlusion,axis=0)
			#accumulate subject-level occlusion map
            if targets==0:
               print('CN')
               occ_CN = np.concatenate([occ_CN,occlusion_map],0)

            if targets==1:
               print('AD')
               occ_AD = np.concatenate([occ_AD,occlusion_map],0)
    
    #compute and save group-level occlusion map of each fold
    occ_AD = np.mean(occ_AD,axis=0)
    np.save('_{}.npy'.format(ki),occ_AD)
    occ_AD = ndimage.zoom(occ_AD, occ_stride, order=3)
    plot_slices(x_train, overlay=occ_AD, overlay_cmap=mycmap,_class=1,iteration=ki)
	occ_all = np.concatenate([occ_all,occ_AD],0)

    occ_CN = np.mean(occ_CN,axis=0)
    np.save('_{}.npy'.format(ki),occ_CN)
    occ_CN = ndimage.zoom(occ_CN, occ_stride, order=3)
    plot_slices(x_train, overlay=occ_CN, overlay_cmap=mycmap,_class=0,iteration=ki)
	occ_all = np.concatenate([occ_all,occ_CN],0)	

#compute and save group-level occlusion map of all the subjects, can give different weight to each class, here we give the same weight for each class
occ_all = np.mean(occ_all,axis=0)
np.save(' .npy',occ_all)

