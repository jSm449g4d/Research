#functionalAPI_test
#main.py -> functionalAPI_ize ->mode seeking
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

def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
    if len(physical_devices)==0:print("GPU_failed!")
    return len(physical_devices)
tf_ini();

def U_INCEPTION_TYSYACHA(input_shape=(None,None,3,)):
    mod=mod_inp = Input(shape=input_shape)
    
    mod_2=Conv2D(64,4,4,padding="same")(mod)
    mod_2=Conv2D(64,3,padding="same",activation="relu")(mod_2)
    mod_2=Dropout(0.2)(mod_2)
    mod_2=Conv2D(64,3,padding="same",activation="relu")(mod_2)
    mod_2=Dropout(0.2)(mod_2)
    mod_2=Conv2DTranspose(3,4,4,padding="same")(mod_2)
    
    mod_1=Conv2D(64,2,2,padding="same")(mod)
    mod_1=Conv2D(64,3,padding="same",activation="relu")(mod_1)
    mod_1=Dropout(0.2)(mod_1)
    mod_1=Conv2D(64,3,padding="same",activation="relu")(mod_1)
    mod_1=Dropout(0.2)(mod_1)
    mod_1=Conv2DTranspose(3,2,2,padding="same")(mod_1)
    
    mod_0=Conv2D(64,1,padding="same")(mod)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Dropout(0.2)(mod_0)
    mod_0=Conv2D(64,3,padding="same",activation="relu")(mod_0)
    mod_0=Dropout(0.2)(mod_0)
    mod_0=Conv2D(3,1,padding="same")(mod_0)
    
    mod+=mod_0+mod_1+mod_2
    
    return keras.models.Model(inputs=mod_inp, outputs=mod)

class c2c():
    def __init__(self,dim=4):
        self.dim=dim
        return
    def __call__(self,mod):
        with tf.name_scope("c2c"):
            mod_1=Conv2D(self.dim,3,padding="same",activation="relu")(mod)
            mod_1=Dropout(0.2)(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same",activation="relu")(mod_1)
            mod_1=Dropout(0.2)(mod_1)
            mod=Activation("relu")(mod+mod_1)
        return mod
    
def DIS(input_shape=(128,128,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod=Conv2D(64,2,2,padding="same")(mod)
    mod=c2c(64)(mod)
    mod=Conv2D(64,2,2,padding="same")(mod)
    mod=c2c(64)(mod)
    mod=Conv2D(64,2,2,padding="same")(mod)
    mod=c2c(64)(mod)
    mod=Conv2D(64,2,2,padding="same")(mod)
    mod=c2c(64)(mod)
    mod=Flatten()(mod)
    mod=Dense(1,activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)
    
class gan():
    def __init__(self,trials=[],dim=(128,128,3)):
        self.gen=U_INCEPTION_TYSYACHA()
        self.dis=DIS()
    def pred(self,batch=4):
        return self.gen(np.random.rand(batch,self.dim).astype(np.float32))
    def train(self,input_x=[],output_y=[],epoch=1000,batch=16,save="./save.h5"):
        optimizer = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
        optimizerA = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
        optimizerB = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
        self.gen.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy)
        self.dis.compile(optimizer = optimizer,
                          loss=keras.losses.binary_crossentropy)
        self.gen.summary()
        self.dis.summary()
        
        try:self.gen.load_weights(os.path.join(args.outdir,"disw.h5"))
        except:print("\nCannot_use_savedata...")
        try:self.dis.load_weights(os.path.join(args.outdir,"genw.h5"))
        except:print("\nCannot_use_savedata...")
                
        ones=np.ones(batch).astype(np.float32)        
        zeros=np.zeros(batch).astype(np.float32)
        
        
        for i in range(epoch):
            gentrainByDiscNumber=0;# for Log
            with tqdm(total=input_x.shape[0]) as pbar:
                
                while pbar.n+batch<pbar.total:
                    datum=output_y[pbar.n:pbar.n+batch];label=input_x[pbar.n:pbar.n+batch];
                    
                    gentrainByDiscFlag=1;
                    #grad,_ = tf.clip_by_global_norm(grad, 15)
                
                    with tf.GradientTape() as tape:
                        dis=self.dis(self.gen(label))
                        dis=keras.losses.binary_crossentropy(zeros,dis)
                        if dis.numpy().mean()>0.2:
                            gentrainByDiscFlag=0
                        dis=tf.reduce_mean(dis) 
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        optimizerA.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        dis=self.dis(datum)
                        dis=keras.losses.binary_crossentropy(ones,dis)
                        if dis.numpy().mean()>0.2:
                            gentrainByDiscFlag=0
                        dis=tf.reduce_mean(dis) 
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        optimizerA.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    if(gentrainByDiscFlag==1):
                        gentrainByDiscNumber+=batch
                        with tf.GradientTape() as tape:
                            dis_gen=self.dis(self.gen(label))
                            dis_gen=keras.losses.binary_crossentropy(ones,dis_gen)
                            dis_gen=tf.reduce_mean(dis_gen) 
                            grad=tape.gradient(dis_gen,self.gen.trainable_variables) 
                            optimizerB.apply_gradients(zip(grad,self.gen.trainable_variables))
                            del tape
                        
                    # tuzyo
                    with tf.GradientTape() as tape:
                        gen=self.gen(label)
                        gen=keras.losses.mse(datum,gen)
                        grad=tape.gradient(gen,self.gen.trainable_variables) 
                        optimizerB.apply_gradients(zip(grad,self.gen.trainable_variables))
                        del tape
                    
                    pbar.update(batch)
                    
            print("\n ke(fake,tru):",
                  tf.reduce_mean(keras.losses.binary_crossentropy (zeros,self.dis(self.gen(input_x[:batch])))).numpy(),
                  tf.reduce_mean(keras.losses.binary_crossentropy (ones,self.dis(output_y[:batch]))).numpy())
            print("gentrainByDiscNumber:",gentrainByDiscNumber)
                    
                    
            self.gen.save(save)
        
def train():
    limitDataSize=min([args.limit_data_size,len(ffzk(args.train_input))])
    x_train=img2np(ffzk(args.train_input)[:limitDataSize]*args.number_of_trainadd,img_len=128)
    y_train=img2np(ffzk(args.train_output)[:limitDataSize]*args.number_of_trainadd,img_len=128)
    
    gans=gan()
    gans.train(x_train,y_train,(args.number_of_backprops//args.limit_data_size)//args.number_of_trainadd,
               batch=args.batch,save=args.save)
    
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
parser.add_argument('-b', '--batch' ,default=4,type=int)
parser.add_argument('-nob', '--number_of_backprops' ,default=100000,type=int)
parser.add_argument('-lds', '--limit_data_size' ,default=100,type=int)
parser.add_argument('-noa', '--number_of_trainadd' ,default=100,type=int)
parser.add_argument('-s', '--save' ,default="./saves/test4.h5")
parser.add_argument('-o', '--outdir' ,default="./outputs/test4")
parser.add_argument('-logdir', '--TB_logdir' ,default="./logs/test4")
args = parser.parse_args()

if __name__ == "__main__":
    if (args.role=="train"):
        train()
    test()
    