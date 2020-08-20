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

os.makedirs("./div2k_learn/train_y",exist_ok=True)
os.makedirs("./div2k_learn/train_4",exist_ok=True)
os.makedirs("./div2k_learn/train_Up4",exist_ok=True)
os.makedirs("./div2k_learn/train_normal",exist_ok=True)
os.makedirs("./div2k_learn/train_fail",exist_ok=True)
os.makedirs("./div2k_learn/train_canny",exist_ok=True)
os.makedirs("./div2k_learn/train_cubic4",exist_ok=True)
os.makedirs("./div2k_learn/test_y",exist_ok=True)
os.makedirs("./div2k_learn/test_4",exist_ok=True)
os.makedirs("./div2k_learn/test_Up4",exist_ok=True)
os.makedirs("./div2k_learn/test_normal",exist_ok=True)
os.makedirs("./div2k_learn/test_fail",exist_ok=True)
os.makedirs("./div2k_learn/test_canny",exist_ok=True)
os.makedirs("./div2k_learn/test_cubic4",exist_ok=True)

lfw=ffzk("./DIV2K_train_HR")
sample=10000000
#lfw=ffzk("./DIV2K_valid_HR")
#sample=0

dSize=128

for i,dirs in enumerate(lfw):
    img=cv2.imread(dirs)
    outputfolder="./div2k_learn/train_"
    if i>sample:
        outputfolder="./div2k_learn/test_"        
        
    cv2.imwrite(outputfolder+"y/"+str(i)+".png",
                cv2.resize(img, dsize=(dSize, dSize)))
    
    Dn=cv2.resize(img, dsize=(int(dSize//4), int(dSize//4)))
    cv2.imwrite(outputfolder+"4/"+str(i)+".png",Dn)
    cv2.imwrite(outputfolder+"Up4/"+str(i)+".png",
                cv2.resize(Dn, dsize=(dSize, dSize)))
    cv2.imwrite(outputfolder+"cubic4/"+str(i)+".png",
                cv2.resize(Dn, dsize=(dSize, dSize), interpolation=cv2.INTER_CUBIC))
    
    img_normal=cv2.resize(img ,(dSize,dSize))+np.random.normal(0, 64, (dSize, dSize,3))
    img_normal=(np.clip(img_normal,0,255)).astype(np.uint8)
    cv2.imwrite(outputfolder+"normal/"+str(i)+".png",
                cv2.resize(img_normal, dsize=(dSize, dSize)))
        
    img_fail=cv2.resize(img ,(dSize,dSize))
    cv2.circle(img_fail, (random.randrange(0, dSize), random.randrange(0, dSize)),
               random.randrange(int(dSize//6), int(dSize//3)), (0, 0, 0),-1)
    img_fail=(np.clip(img_fail,0,255)).astype(np.uint8)
    cv2.imwrite(outputfolder+"fail/"+str(i)+".png",
                cv2.resize(img_fail, dsize=(dSize, dSize)))
        
    img_edges = cv2.Canny(img,50,100)
    cv2.imwrite(outputfolder+"canny/"+str(i)+".png",img_edges)
    
        
