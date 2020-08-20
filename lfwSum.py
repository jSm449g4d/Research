import os
import numpy as np
import cv2
import shutil
import random

def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array


lfw=ffzk("./lfw_learn/test_y/")
dSize=128

img=cv2.imread(lfw[0])/(len(lfw)+1)

for i,dirs in enumerate(lfw):
    img+=cv2.imread(dirs)/(len(lfw)+1)
          
        
cv2.imwrite("./sum.png",
            cv2.resize(img, dsize=(dSize, dSize)))
    
    