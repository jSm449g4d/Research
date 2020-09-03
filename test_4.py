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

def UNET_EZ(input_shape=(None,None,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod=Conv2D(64,3,padding="same",activation="relu")(mod)
    mod_1=mod
    mod_1=Conv2D(64,2,2,padding="same",activation="relu")(mod_1)
    mod_2=mod_1
    mod_2=Conv2D(64,2,2,padding="same",activation="relu")(mod_2)
    mod_2=Conv2D(64,3,padding="same",activation="relu")(mod_2)
    mod_2=Conv2DTranspose(64,3,2,padding="same",activation="relu")(mod_2)
    mod_1=mod_1+mod_2
    mod_1=Conv2D(64,3,padding="same",activation="relu")(mod_1)
    mod_1=Conv2DTranspose(64,3,2,padding="same",activation="relu")(mod_1)
    mod=mod+mod_1
    mod=Conv2D(64,3,padding="same",activation="relu")(mod)
    mod=Conv2D(64,3,padding="same",activation="relu")(mod)
    mod=Conv2D(3,3,padding="same")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def Discriminator(input_shape=(None,None,3,)):
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train_step(gen,disc,input_x,true_y):
    def generator_loss(fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      output_y = gen(input_x, training=True)

      real_output = disc(input_x, training=True)
      fake_output = disc(output_y, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999).apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999).apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

def train():
    limitDataSize=min([args.limit_data_size,len(ffzk(args.train_input))])
    x_train=img2np(ffzk(args.train_input)[:limitDataSize],img_len=128)
    y_train=img2np(ffzk(args.train_output)[:limitDataSize],img_len=128)
    x_test=img2np(ffzk(args.pred_input),img_len=128)
    y_test=img2np(ffzk(args.pred_output),img_len=128)
    #[:10]*1000
    
    gen=UNET_EZ()
    disc=Discriminator()
    
    for epoch in range(args.epoch):
        for x,y in zip(x_train,y_train):
            train_step(gen,disc,x,y)
    
    gen.save(args.save)
    
def test():
    model = keras.models.load_model(args.save)
    os.makedirs(args.outdir,exist_ok=True)
    dataset=ffzk(args.pred_input)
    for i,dataX in enumerate(dataset):
        predY=model.predict(img2np([dataX],img_len=128))
        tf2img(predY,args.outdir,name=os.path.basename(dataX))

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--role' ,default="train")
parser.add_argument('-ti', '--train_input' ,default="./datasets/div2k_srlearn/train_cubic8")
parser.add_argument('-to', '--train_output' ,default="./datasets/div2k_srlearn/train_y")
parser.add_argument('-pi', '--pred_input' ,default='./datasets/div2k_srlearn/test_cubic8')
parser.add_argument('-po', '--pred_output' ,default='./datasets/div2k_srlearn/test_y')
parser.add_argument('-b', '--batch' ,default=2,type=int)
parser.add_argument('-e', '--epoch' ,default=10,type=int)
parser.add_argument('-lds', '--limit_data_size' ,default=10000,type=int)
parser.add_argument('-s', '--save' ,default="./saves/test4.h5")
parser.add_argument('-o', '--outdir' ,default="./outputs/test4")
parser.add_argument('-logdir', '--TB_logdir' ,default="./logs/test4")
args = parser.parse_args()

if __name__ == "__main__":
    if (args.role=="train"):
        train()
    test()
