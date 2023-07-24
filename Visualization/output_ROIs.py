import numpy as np
import nibabel as nib
import os
import pandas as pd
from scipy import ndimage 


#load heatmap
OCC = np.load("  .npy")
occ_stride = 5
#interpolation
OCC = ndimage.zoom(OCC, occ_stride, order=3)

D = 218
H = 182
W = 182
#set hyperparameter
patch_size = 30
step = 20
list = []
list_co = []


#rank ROIS based on heatmap
for z in range(0,D-patch_size+1,step):         
   for y in range(0,H-patch_size+1,step):
      for x in range(0,W-patch_size+1,step):
                 locx = OCC[x:x+patch_size,z:z+patch_size,y:y+patch_size]
                 weigh_sum = np.sum(locx)
                 list_co.append((x,z,y))
                 list.append(weigh_sum)
                                
print(np.argsort(list))
#save the location of ROIs
with open('ROIs.txt','w') as f:
   for i in np.argsort(list)[-20:]:
      f.write(str(list_co[i]))
















