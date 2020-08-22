from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
from skimage import data, img_as_float
import cv2
import time
import os
import numpy as np
from matplotlib import pylab as plt
import time
import math

def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

def img2np(dir=[],img_len=128):
    img=[]
    for x in dir:
        try:img.append(cv2.imread(x))
        except:continue
        if img_len!=0:img[-1]=cv2.resize(img[-1],(img_len,img_len))
        elif img[-1].shape!=img[0].shape:img.pop(-1);continue#Leave only the same shape
        img[-1] = img[-1].astype(np.float32)/ 256
    return np.stack(img, axis=0)

preds=[]

test=ffzk(os.path.join("./", 'div2k_srlearn/test_y'))
preds.append(ffzk(os.path.join("./", 'div2k_srlearn/test_cubic4')))
preds.append(ffzk(os.path.join("./", 'out1')))
preds.append(ffzk(os.path.join("./", 'out2')))
preds.append(ffzk(os.path.join("./", 'out3')))
# preds.append(ffzk(os.path.join("./", 'outSrcnndrei')))
preds.append(ffzk(os.path.join("./", 'outSrcnnvier')))
preds.append(ffzk(os.path.join("./", 'outSrcnnfunf')))



# test=ffzk(os.path.join("./", 'div2k_srlearn/test_y'))
# preds.append(ffzk(os.path.join("./", 'div2k_srlearn/test_normal')))
# preds.append(ffzk(os.path.join("./", 'outSrcnnNormal')))
# preds.append(ffzk(os.path.join("./", 'outSrcnnzweiNormal')))
# preds.append(ffzk(os.path.join("./", 'div2k_srlearn/test_normal')))
# preds.append(ffzk(os.path.join("./", 'div2k_srlearn/test_median')))
# preds.append(ffzk(os.path.join("./", 'div2k_srlearn/test_gaussb')))

max_sample_size=min([1000,len(test)])

for i in range(len(preds)):
    pnsrS=0.;ssimS=0.;meS=0.;mseS=0.
    for ii in range(max_sample_size):
        img1 = cv2.imread(test[ii])
        img2 = cv2.imread(preds[i][ii])
#        pnsrS+=psnr(img1, img2)
        ssimS+=ssim(img1, img2, multichannel=True)
        meS+=mse(img1.flatten(),img2.flatten())**2.
#    pnsrS/=max_sample_size;
    ssimS/=max_sample_size;
    mseS=math.sqrt(meS/max_sample_size)
#    print("PSNR",pnsrS)
    print("SSIM",ssimS)
    print("PSNR",10.*math.log10((255.**2)/mseS))
    print("=====^",i,"^=====")
    
#for i in range(len(preds)):
#    img1 = cv2.imread(test[77])
#    img2 = cv2.imread(preds[i][77])
#    print("PSNR",psnr(img1, img2))
#    print("SSIM",ssim(img1, img2, multichannel=True))
#    print("=====^",i,"^=====")
