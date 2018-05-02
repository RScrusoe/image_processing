import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.io as io
import matplotlib.colors as colors
from skimage import *
from matplotlib import cm
import scipy.io as sio



#variables here
disparity_range = 50
block_size = 3

imgL = io.imread('left1.png')
imgR = io.imread('right1.png')

grayL = color.rgb2gray(imgL)
grayR = color.rgb2gray(imgR)

out = np.zeros(grayL.shape)
imgHeight,imgWidth = grayL.shape[0],grayL.shape[1]
print(grayL.shape)

for i in range(imgHeight):
    rmin = max(0,i-block_size-1)
    rmax = min(imgHeight-1, i + block_size - 1)
    for j in range(imgWidth):
        cmin = max(0,j-block_size-1)
        cmax = min (imgWidth-1,j+block_size-1)
        
        #Offset ie number of pixels till which we need to search
        # We set it zero for left for "Cones" dataset
        min_offset = 0
        max_offset = min(disparity_range-1,imgWidth-cmax-1)

        #Select block from right image
        tmp = grayR[rmin:rmax,cmin:cmax]

        #total possible blocks to compare :
        nblocks = max_offset - min_offset + 1

        #blockwise differences
        block_diff = np.zeros((nblocks,1))
        for k in range(min_offset,max_offset+1):
            curr_block = grayL[rmin:rmax,(cmin+k):(cmax+k)]
            index = k - min_offset + 1 
            block_diff[index-1,0] = sum(sum(abs(tmp-curr_block)))
        
        sortex_indices = np.argsort(block_diff.ravel())
        best_index = sortex_indices[0]

        #Calaculate Disparity
        disparity = best_index + min_offset 
        #out[i,j] = disparity
        if (best_index == 0) or (best_index == nblocks-1):
            out[i,j] = float(disparity)
        else:
            #  Closest matching neighboring blocks 
            B1 = float(block_diff[best_index-1])
            B2 = float(block_diff[best_index])
            B3 = float(block_diff[best_index+1])

            out[i,j] = float(disparity) - (0.5*(B3-B1)/(B1 - (2.0*B2) + B3) )


    print(i)

print(out.shape)
sio.savemat('out1.mat', {'vect':out})

print('done!')
# print(out.shape)

out = out.astype(np.uint8)
# np.savetxt("foo.csv", out)

# # out[0] = 255.0*(out[0] - np.min(out[0]))/np.max(out[0])
# # out[1] = 255.0*(out[1] - np.min(out[1]))/np.max(out[1])
# # out[2] = 255.0*(out[2] - np.min(out[2]))/np.max(out[2])
# out = out.astype(np.uint8)

f1 = plt.figure()
plt.imshow(out,cmap=cm.coolwarm)
plt.show()
# print(out)
# print(np.max(out))

# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# # print(np.mean(out))

# io.imshow(out)
# io.show()