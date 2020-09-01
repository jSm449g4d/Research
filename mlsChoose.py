import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import random

# https://data.nasa.gov/Space-Science/Mars-surface-image-Curiosity-rover-labeled-data-se/cjex-ucks
def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

lfw=ffzk("./calibrated")

# myimg = cv2.imread("./msl/5845.png")
# avg_color_per_row = np.average(myimg, axis=0)
# avg_color = np.average(avg_color_per_row, axis=0)
# print(avg_color)

os.makedirs("./msl",exist_ok=True)
os.makedirs("./mslReject",exist_ok=True)
for i,v in enumerate(lfw):
    myimg = cv2.imread(v)
    
    _height, _width, _channels =myimg.shape[:3]
    if(_height<128 or _width<128):
        continue
    
    avg_color = np.average( np.average(myimg, axis=0), axis=0)
    if(avg_color[2]>avg_color[1]*1.2 and avg_color[2]>avg_color[0]*1.2 and avg_color[2]>120):
        cv2.imwrite("./msl/"+str(random.randrange(0, 1000000))+".png",myimg)
    else:
        cv2.imwrite("./mslReject/"+str(i)+".png",myimg)
