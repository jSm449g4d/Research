import cv2
from matplotlib import pyplot as plt
import numpy as np
from spmimage.decomposition import KSVD
import glob
import multiprocessing
import os
import time

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

def tf2img(tfs,_dir="./",name="",epoch=0,ext=".png"):
    os.makedirs(_dir, exist_ok=True)
    if type(tfs)!=np.ndarray:tfs=tfs.numpy()
    tfs=(np.clip(tfs,0.0, 1.0)*256).astype(np.uint8)
    for i in range(tfs.shape[0]):
        cv2.imwrite(os.path.join(_dir,name),tfs[i])
    
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

    #start = time.time()
    #process_time = time.time() - start;print("stack",process_time)
#train

start = time.time()
y_train=img2np(ffzk(os.path.join("./", 'div2k_srlearn/train_cubic4'))[:2])#.reshape(-1,8*8*3)
y_train=splitHVS(y_train)
process_time = time.time() - start;print("A",process_time);start = time.time()

ksvd = KSVD(n_components = 32, transform_n_nonzero_coefs = None ,n_jobs=multiprocessing.cpu_count()-1)
ksvd.fit(y_train)

process_time = time.time() - start;print("B",process_time);start = time.time()

#pred
os.makedirs(os.path.join("./", 'outKSVD'),exist_ok=True)
dataset=ffzk(os.path.join("./", './div2k_srlearn/test_cubic4'))
for i,dataX in enumerate(dataset):
    X = ksvd.transform(splitHVS(img2np([dataX],img_len=128)))
    predY=stackHVS(np.dot(X, ksvd.components_))
    tf2img(predY,os.path.join("./", 'outKSVD'),name=os.path.basename(dataX))

process_time = time.time() - start;print("C",process_time);start = time.time()
