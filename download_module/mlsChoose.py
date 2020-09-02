import cv2
import numpy as np
import os
import random

# Mars surface image (Curiosity rover) labeled data set
def ffzk(input_dir):
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

lfw=ffzk(os.path.join("./datasets/calibrated"))

os.makedirs("./datasets/mls",exist_ok=True)
os.makedirs("./datasets/mls",exist_ok=True)
for i,v in enumerate(lfw):
    myimg = cv2.imread(v)
    
    _height, _width, _channels =myimg.shape[:3]
    if(_height<128 or _width<128):
        continue
    
    avg_color = np.average( np.average(myimg, axis=0), axis=0)
    if(avg_color[2]>avg_color[1]*1.2 and avg_color[2]>avg_color[0]*1.2 and avg_color[2]>120):
        cv2.imwrite("./datasets/mls/"+str(random.randrange(0, 1000000))+".png",myimg)
    else:
        cv2.imwrite("./datasets/mls/"+str(i)+".png",myimg)
