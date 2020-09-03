import cv2
from matplotlib import pyplot as plt
import numpy as np
from spmimage.decomposition import KSVD
import glob
import multiprocessing
import os
import time
from util import ffzk,img2np,tf2img

def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array
    
def splitHV(imgs,size=128,spl=16):
    v_size = imgs[0].shape[0] // size * size
    h_size = imgs[0].shape[1] // size * size
    imgs = [i[:v_size, :h_size] for i in imgs]
    v_split = imgs[0].shape[0] // size
    h_split = imgs[0].shape[1] // size
    out_imgs = []
    [[out_imgs.extend(np.hsplit(h_img, spl))
      for h_img in np.vsplit(img, spl)] for img in imgs]
    return np.stack(out_imgs)
def splitHVS(imgs,size=128,spl=8):
    return splitHV(imgs,size,spl).reshape(-1,size*size*3//(spl*spl))

def stackHV(imgs,spl=16):
    out_imgs=[]
    for i_img in range(len(imgs)//(spl)):
        out_imgs.append(np.hstack(imgs[spl*i_img:spl*(i_img+1)]))
    imgs=out_imgs;out_imgs=[]
    for i_img in range(len(imgs)//(spl)):
        out_imgs.append(np.vstack(imgs[spl*i_img:spl*(i_img+1)]))
    return np.stack(out_imgs)
def stackHVS(imgs,size=128,spl=8):
    imgs=imgs.reshape(-1,size//(spl),size//(spl),3)
    return stackHV(imgs,spl).reshape(-1,size,size,3)

start = time.time()
y_train=img2np(ffzk('./datasets/div2k_srlearn/train_y')[:10])#.reshape(-1,8*8*3)
y_train=splitHVS(y_train)
process_time = time.time() - start;print("A",process_time);start = time.time()

ksvd = KSVD(n_components = 32, transform_n_nonzero_coefs = None ,n_jobs=multiprocessing.cpu_count()-1)
ksvd.fit(y_train)

process_time = time.time() - start;print("B",process_time);start = time.time()

#pred
os.makedirs('./outputs/ksvd',exist_ok=True)
dataset=ffzk('./datasets/div2k_srlearn/test_cubic8')[:1000]
for i,dataX in enumerate(dataset):
    X = ksvd.transform(splitHVS(img2np([dataX],img_len=128)))
    predY=stackHVS(np.dot(X, ksvd.components_))
    tf2img(predY,'./outputs/ksvd',name=os.path.basename(dataX))

process_time = time.time() - start;print("C",process_time);start = time.time()
