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


def VDSR(input_shape=(None,None,3,)):
    mod=mod_inp = Input(shape=input_shape)
    
    #,activity_regularizer=keras.regularizers.L2(0.0001)
    #for _ in range(1):
    #    mod_p=Conv2D(64,3,padding="same",activation="relu",activity_regularizer=keras.regularizers.L2(0.0001))(mod_p)
    mod_0=Conv2D(64,1,padding="same")(mod)
    
    # 4Conv
    mod_0=LayerNormalization()(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    # 4Conv
    mod_0=LayerNormalization()(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    # 4Conv
    mod_0=LayerNormalization()(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    # 4Conv
    mod_0=LayerNormalization()(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    # 4Conv
    mod_0=LayerNormalization()(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    
    mod_0=Conv2D(3,1,padding="same")(mod_0)
    mod+=mod_0
    
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def train():
    limitDataSize=min([args.limit_data_size,len(ffzk(args.train_input))])
    x_train=img2np(ffzk(args.train_input)[:limitDataSize]*(10000//args.limit_data_size),img_len=128)
    y_train=img2np(ffzk(args.train_output)[:limitDataSize]*(10000//args.limit_data_size),img_len=128)
    x_test=img2np(ffzk(args.pred_input),img_len=128)
    y_test=img2np(ffzk(args.pred_output),img_len=128)
    
    model=VDSR()
    model.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999),
                  loss=keras.losses.mean_squared_error)#keras.losses.mean_squared_error
    model.summary()
    cbks=[]
    if(args.TB_logdir!=""):
        cbks=[keras.callbacks.TensorBoard(log_dir=args.TB_logdir, histogram_freq=1)]
    
    model.fit(x_train, y_train,epochs=(args.number_of_backprops//args.limit_data_size)//(10000//args.limit_data_size),
              batch_size=args.batch,validation_data=(x_test, y_test),callbacks=cbks)
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
parser.add_argument('-ti', '--train_input' ,default="./datasets/div2k_srlearn/train_cubic4")
parser.add_argument('-to', '--train_output' ,default="./datasets/div2k_srlearn/train_y")
parser.add_argument('-pi', '--pred_input' ,default='./datasets/div2k_srlearn/test_cubic4')
parser.add_argument('-po', '--pred_output' ,default='./datasets/div2k_srlearn/test_y')
parser.add_argument('-b', '--batch' ,default=1,type=int)
parser.add_argument('-nob', '--number_of_backprops' ,default=100000,type=int)
parser.add_argument('-lds', '--limit_data_size' ,default=10000,type=int)
parser.add_argument('-s', '--save' ,default="./saves/vdsr4.h5")
parser.add_argument('-o', '--outdir' ,default="./outputs/vdsr4")
parser.add_argument('-logdir', '--TB_logdir' ,default="./logs/vdsr4")
args = parser.parse_args()

if __name__ == "__main__":
    if (args.role=="train"):
        train()
    test()
