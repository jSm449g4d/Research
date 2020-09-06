import cv2
from matplotlib import pyplot as plt
import numpy as np
from spmimage.decomposition import KSVD
import multiprocessing
import os
import time
import pickle
from util import ffzk,img2np,tf2img
import argparse
    
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
def splitHVS(imgs,size=128,spl=16):
    return splitHV(imgs,size,spl).reshape(-1,size*size*3//(spl*spl))

def stackHV(imgs,spl=16):
    out_imgs=[]
    for i_img in range(len(imgs)//(spl)):
        out_imgs.append(np.hstack(imgs[spl*i_img:spl*(i_img+1)]))
    imgs=out_imgs;out_imgs=[]
    for i_img in range(len(imgs)//(spl)):
        out_imgs.append(np.vstack(imgs[spl*i_img:spl*(i_img+1)]))
    return np.stack(out_imgs)
def stackHVS(imgs,size=128,spl=16):
    imgs=imgs.reshape(-1,size//(spl),size//(spl),3)
    return stackHV(imgs,spl).reshape(-1,size,size,3)

def train():
    start = time.time()
    limitDataSize=min([args.limit_data_size,len(ffzk(args.train_input))])
    y_train=img2np(ffzk(args.train_output)[:limitDataSize])#.reshape(-1,8*8*3)
    y_train=splitHVS(y_train,spl=args.image_split)
    process_time = time.time() - start;print("A",process_time);start = time.time()

    ksvd = KSVD(n_components =16, transform_n_nonzero_coefs = None ,n_jobs=multiprocessing.cpu_count()-1)
    ksvd.fit(y_train)
    pickle.dump(ksvd, open(args.save, 'wb'))

    process_time = time.time() - start;print("B",process_time);start = time.time()

def test():
    start = time.time()
    ksvd=pickle.load(open(args.save, 'rb'))
    os.makedirs(args.outdir,exist_ok=True)
    dataset=ffzk(args.pred_input)
    for i,dataX in enumerate(dataset):
        X = ksvd.transform(splitHVS(img2np([dataX],img_len=128),spl=args.image_split))
        predY=stackHVS(np.dot(X, ksvd.components_),spl=args.image_split)
        tf2img(predY,args.outdir,name=os.path.basename(dataX))
        
    process_time = time.time() - start;print("C",process_time);start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--role' ,default="train")
parser.add_argument('-ti', '--train_input' ,default="./datasets/div2k_srlearn/train_cubic4")
parser.add_argument('-to', '--train_output' ,default="./datasets/div2k_srlearn/train_y")
parser.add_argument('-pi', '--pred_input' ,default='./datasets/div2k_srlearn/test_cubic4')
parser.add_argument('-po', '--pred_output' ,default='./datasets/div2k_srlearn/test_y')
parser.add_argument('-lds', '--limit_data_size' ,default=100,type=int)
parser.add_argument('-spl', '--image_split' ,default=16,type=int)
parser.add_argument('-s', '--save' ,default="./saves/ksvd5.pickle")
parser.add_argument('-o', '--outdir' ,default='./outputs/ksvd5')
args = parser.parse_args()

if __name__ == "__main__":
    if (args.role=="train"):
        train()
    test()