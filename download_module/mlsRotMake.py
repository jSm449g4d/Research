import os
import numpy as np
import cv2
import random
from util import ffzk
#https://data.vision.ee.ethz.ch/cvl/DIV2K/

dSize=128

lfw=ffzk("./datasets/mls")[1000:]
sample=10000
outputfolder="./datasets/mls_srlearn/train_"

for iv in [0,1]:
    if iv==1:
        lfw=ffzk("./datasets/msl")[:1000]
        sample=1000
        outputfolder="./datasets/mls_srlearn/test_" 
        
    # Reuse
    os.makedirs(outputfolder+"y",exist_ok=True)
    os.makedirs(outputfolder+"4",exist_ok=True)
    os.makedirs(outputfolder+"cubic4",exist_ok=True)
    os.makedirs(outputfolder+"cubic4normal",exist_ok=True)
    os.makedirs(outputfolder+"normal",exist_ok=True)
    os.makedirs(outputfolder+"gaussb",exist_ok=True)
    os.makedirs(outputfolder+"gray",exist_ok=True)
    os.makedirs(outputfolder+"cubic8",exist_ok=True)

    for i in range(sample):
        dirs=random.choice(lfw)
        
        img=cv2.imread(dirs)
        _height, _width, _channels =img.shape[:3]
                       
        _height=random.randrange(0, int(_height-dSize))
        _width=random.randrange(0, int(_width-dSize))
        img=img[_height:_height+dSize,_width:_width+dSize]
        img=np.rot90(img,random.choice([0,1,2,3]))
        
        cv2.imwrite(outputfolder+"y/"+str(i)+".png",cv2.resize(img, dsize=(dSize, dSize)))
        
        Dn=cv2.resize(cv2.resize(img, dsize=(int(dSize//4), int(dSize//4))), dsize=(dSize, dSize), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(outputfolder+"cubic4/"+str(i)+".png",Dn)
                
        img_normal=cv2.resize(img ,(dSize,dSize))+np.random.normal(0, 64, (dSize, dSize,3))
        img_normal=np.clip(img_normal,0,255).astype(np.uint8)
        cv2.imwrite(outputfolder+"normal/"+str(i)+".png",img_normal)
        cv2.imwrite(outputfolder+"gaussb/"+str(i)+".png",cv2.GaussianBlur(img_normal,(5,5),0))
        
        img_normal=Dn+np.random.normal(0, 64, (dSize, dSize,3))
        img_normal=np.clip(img_normal,0,255).astype(np.uint8)
        cv2.imwrite(outputfolder+"cubic4normal/"+str(i)+".png",img_normal)
        
        # extra
        cv2.imwrite(outputfolder+"gray/"+str(i)+".png",cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        Dn=cv2.resize(cv2.resize(img, dsize=(int(dSize//8), int(dSize//8))), dsize=(dSize, dSize), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(outputfolder+"cubic8/"+str(i)+".png",Dn)
        