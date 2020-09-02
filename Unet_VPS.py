import os
import sys
import tensorflow as tf
import numpy as np
import cv2

import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from util import ffzk,img2np,tf2img

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

if __name__ == "__main__":
    model = keras.models.load_model("Unet_VPS.model")
    predY=model.predict(img2np(["./input.png"]))
    tf2img(predY,"./",name="output.png")
