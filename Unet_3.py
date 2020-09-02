import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization,\
Lambda,Multiply,GlobalAveragePooling2D,LeakyReLU,PReLU,BatchNormalization,\
Conv2DTranspose,MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from tqdm import tqdm
import argparse
from util import ffzk,img2np,tf2img

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

class UNET_EZ_EGSRT():#efficient  grid  size reduction  techniques
    def __init__(self,dim=32):
        self.dim=dim;
        return
    def __call__(self,mod):
        with tf.name_scope("UnetEGSRT"):
            mod=Conv2D(self.dim,3,padding="same",activation="relu")(mod)
            mod=Dropout(0.2)(mod)
            mod_1=mod
            mod_1=Conv2D(self.dim,2,2,padding="same",activation="relu")(mod_1)
            mod_1=Dropout(0.2)(mod_1)
            mod_2=mod_1
            mod_2=Conv2D(self.dim,2,2,padding="same",activation="relu")(mod_2)
            mod_2=Dropout(0.2)(mod_2)
            mod_2=Conv2D(self.dim,3,padding="same",activation="relu")(mod_2)
            mod_2=UpSampling2D(2)(mod_2)
            mod_1=mod_1+mod_2
            mod_1=Conv2D(self.dim,3,padding="same",activation="relu")(mod_1)
            mod_1=UpSampling2D(2)(mod_1)
            mod=mod+mod_1
            mod=Conv2D(self.dim,3,padding="same",activation="relu")(mod)
        return mod
    
def UNET_EZ_BEN(input_shape=(128,128,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod_1=UNET_EZ_EGSRT(48)(mod)
    mod_2=UNET_EZ_EGSRT(48)(mod)
    mod_3=UNET_EZ_EGSRT(48)(mod)
    mod_4=UNET_EZ_EGSRT(48)(mod)
    mod=mod_1+mod_2+mod_3+mod_4
    mod=Conv2D(3,3,padding="same")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)
    
def train():
    limitDataSize=min([args.limit_data_size,len(ffzk(args.train_input))])
    x_train=img2np(ffzk(args.train_input)[:limitDataSize],img_len=128)
    y_train=img2np(ffzk(args.train_output)[:limitDataSize],img_len=128)
    x_test=img2np(ffzk(args.pred_input),img_len=128)
    y_test=img2np(ffzk(args.pred_output),img_len=128)
    #[:10]*1000
    
    model=UNET_EZ_BEN()
    model.compile(optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),
                  loss=keras.losses.mean_squared_error)#keras.losses.mean_squared_error
    model.summary()
    cbks=[]
    if(args.TB_logdir!=""):
        cbks=[keras.callbacks.TensorBoard(log_dir=args.TB_logdir, histogram_freq=1)]
    
    model.fit(x_train, y_train,epochs=args.epoch,batch_size=args.batch,validation_data=(x_test, y_test),callbacks=cbks)
    model.save(args.save)
    
def test():
    model = keras.models.load_model(args.save)
    os.makedirs(args.outdir,exist_ok=True)
    dataset=ffzk(args.pred_input)
    for i,dataX in enumerate(dataset):
        predY=model.predict(img2np([dataX],img_len=128))
        tf2img(predY,args.outdir,name=os.path.basename(dataX))

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--role' ,default="train")
parser.add_argument('-ti', '--train_input' ,default="./mls_srlearn/train_cubic4")
parser.add_argument('-to', '--train_output' ,default="./mls_srlearn/train_y")
parser.add_argument('-pi', '--pred_input' ,default='./mls_srlearn/test_cubic4')
parser.add_argument('-po', '--pred_output' ,default='./mls_srlearn/test_y')
parser.add_argument('-b', '--batch' ,default=2,type=int)
parser.add_argument('-e', '--epoch' ,default=20,type=int)
parser.add_argument('-lds', '--limit_data_size' ,default=10000,type=int)
parser.add_argument('-s', '--save' ,default="./model3.h5")
parser.add_argument('-o', '--outdir' ,default="./out3Mls")
parser.add_argument('-logdir', '--TB_logdir' ,default="log3Mls")
args = parser.parse_args()

if __name__ == "__main__":
    if (args.role=="train"):
        train()
    test()
