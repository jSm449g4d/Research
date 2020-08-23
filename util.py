import os
import cv2
import numpy as np
        
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
        img[-1] = img[-1].astype(np.float32)/ 255.
    return np.stack(img, axis=0)

def tf2img(tfs,_dir="./",name="",epoch=0,ext=".png"):
    os.makedirs(_dir, exist_ok=True)
    if type(tfs)!=np.ndarray:tfs=tfs.numpy()
    tfs=np.clip(np.round(tfs*255.),0, 255).astype(np.uint8)
    for i in range(tfs.shape[0]):
        cv2.imwrite(os.path.join(_dir,name),tfs[i])
        