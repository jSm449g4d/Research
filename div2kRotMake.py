import os
import numpy as np
import cv2
import shutil
import random
#https://data.vision.ee.ethz.ch/cvl/DIV2K/
def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

os.makedirs("./div2k_srlearn/train_y",exist_ok=True)
os.makedirs("./div2k_srlearn/train_4",exist_ok=True)
os.makedirs("./div2k_srlearn/train_cubic4",exist_ok=True)
os.makedirs("./div2k_srlearn/train_normal",exist_ok=True)
os.makedirs("./div2k_srlearn/train_median",exist_ok=True)
os.makedirs("./div2k_srlearn/train_gaussb",exist_ok=True)
os.makedirs("./div2k_srlearn/train_bilatr",exist_ok=True)
os.makedirs("./div2k_srlearn/test_y",exist_ok=True)
os.makedirs("./div2k_srlearn/test_4",exist_ok=True)
os.makedirs("./div2k_srlearn/test_cubic4",exist_ok=True)
os.makedirs("./div2k_srlearn/test_normal",exist_ok=True)
os.makedirs("./div2k_srlearn/test_median",exist_ok=True)
os.makedirs("./div2k_srlearn/test_gaussb",exist_ok=True)
os.makedirs("./div2k_srlearn/test_bilatr",exist_ok=True)

lfw=ffzk("./DIV2K_train_HR")
sample=50000
outputfolder="./div2k_srlearn/train_"
# lfw=ffzk("./DIV2K_valid_HR")
# sample=1000
# outputfolder="./div2k_srlearn/test_" 

dSize=128

for i in range(sample):
    dirs=random.choice(lfw)
    
    img=cv2.imread(dirs)
    _height, _width, _channels =img.shape[:3]
                   
    _height=random.randrange(0, int(_height-dSize))
    _width=random.randrange(0, int(_width-dSize))
    img=img[_height:_height+dSize,_width:_width+dSize]
    img=cv2.flip(img,random.choice([-1,0,1]))
    
    cv2.imwrite(outputfolder+"y/"+str(i)+".png",cv2.resize(img, dsize=(dSize, dSize)))
    
    Dn=cv2.resize(img, dsize=(int(dSize//4), int(dSize//4)))
    cv2.imwrite(outputfolder+"4/"+str(i)+".png",Dn)
    cv2.imwrite(outputfolder+"cubic4/"+str(i)+".png",cv2.resize(Dn, dsize=(dSize, dSize), interpolation=cv2.INTER_CUBIC))
       
    img_normal=cv2.resize(img ,(dSize,dSize))+np.random.normal(0, 64, (dSize, dSize,3))
    img_normal=np.clip(img_normal,0,255).astype(np.uint8)
    cv2.imwrite(outputfolder+"normal/"+str(i)+".png",img_normal)
    cv2.imwrite(outputfolder+"median/"+str(i)+".png",cv2.medianBlur(img_normal,7))
    cv2.imwrite(outputfolder+"gaussb/"+str(i)+".png",cv2.GaussianBlur(img_normal,(7,7),0))
    cv2.imwrite(outputfolder+"bilatr/"+str(i)+".png",cv2.bilateralFilter(img_normal,7,100,100))
