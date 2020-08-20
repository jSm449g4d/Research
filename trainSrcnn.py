import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        
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
        cv2.imwrite(os.path.join(_dir,name+"_epoch-num_"+str(epoch)+"-"+str(i)+ext),tfs[i])
    
class Plain():
    def __init__(self,dim=4):#plz define used layers below...
        self.dim=dim
        return
    def __call__(self,mod):#plz add layers below...
        with tf.name_scope("Plain"):
            mod_1=mod
            mod_1=Conv2D(self.dim,3,padding="same",activation="relu")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Activation("relu")(mod)
        return mod
    
class Bottleneck():
    def __init__(self,dim=4):#plz define used layers below...
        self.dim=dim
        return
    def __call__(self,mod):#plz add layers below...
        with tf.name_scope("Bn"):
            mod_1=mod
            mod_1=Conv2D(self.dim//4,1,padding="same",activation="relu")(mod_1)
            mod_1=Conv2D(self.dim//4,3,padding="same",activation="relu")(mod_1)
            mod_1=Conv2D(self.dim,1,padding="same")(mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Activation("relu")(mod)
        return mod
    
def GEN(input_shape=(128,128,3,)):
    mod=mod_inp = Input(shape=input_shape)
    
    mod=Conv2D(64,9,padding="same",activation="relu")(mod)
    mod=Conv2D(64,1,padding="same",activation="relu")(mod)    
    mod=Conv2D(64,1,padding="same",activation="relu")(mod)    

    mod=Conv2D(3,8,padding="same",kernel_regularizer=regularizers.l1(0.0))(mod)
    
    return keras.models.Model(inputs=mod_inp, outputs=mod)

x_train=img2np(ffzk(os.path.join("./", './lfw_learn/train_4'))[:10000]*1,img_len=128)
y_train=img2np(ffzk(os.path.join("./", './lfw_learn/train_y'))[:10000]*1,img_len=128)
x_test=img2np(ffzk(os.path.join("./", './lfw_learn/test_4')),img_len=128)
y_test=img2np(ffzk(os.path.join("./", './lfw_learn/test_y')),img_len=128)
#[:10]*1000
def main():
    
    model=GEN()
    model.compile(optimizer=optimizers.Adam(),
                  loss=keras.losses.mean_squared_error)#keras.losses.mean_squared_error
    model.summary()
    cbks=[keras.callbacks.TensorBoard(log_dir='logsSrcnn', histogram_freq=1)]
    cbks=[]
    
    model.fit(x_train, y_train,epochs=20,validation_data=(x_test, y_test),callbacks=cbks)
    model.save('modelSrcnn.h5')

if __name__ == "__main__":
    tf_ini()
    main()
    