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
Conv2DTranspose,MaxPooling2D,AveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from tqdm import tqdm
import argparse
from util import ffzk,img2np,tf2img
from tensorflow.keras.utils import plot_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

plot_model(
    keras.models.load_model("SR_VPS.h5"),
    to_file="./model.png",
    show_shapes=True,
)