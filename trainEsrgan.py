#functionalAPI_test
#main.py -> functionalAPI_ize ->mode seeking
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm
import argparse
import os
import random


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
    
class c3c():
    def __init__(self,dim=4):#plz define used layers below...
        self.dim=dim
        return
    def __call__(self,mod):#plz add layers below...
        with tf.name_scope("c3c"):
            mod_1=mod
            mod=Conv2D(self.dim,1,padding="same")(mod)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Dropout(0.05)(mod)
            mod=Activation("relu")(mod)
        return mod
    
    

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ResDenseBlock_5C(tf.keras.layers.Layer):
    """Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RDB5C',
                 **kwargs):
        super(ResDenseBlock_5C, self).__init__(name=name, **kwargs)
        # gc: growth channel, i.e. intermediate channels
        self.res_beta = res_beta
        lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            Conv2D, kernel_size=3, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(wd))
        self.conv1 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv2 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv3 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv4 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv5 = _Conv2DLayer(filters=nf, activation=lrelu_f())

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.concat([x, x1], 3))
        x3 = self.conv3(tf.concat([x, x1, x2], 3))
        x4 = self.conv4(tf.concat([x, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], 3))
        return x5 * self.res_beta + x


class ResInResDenseBlock(tf.keras.layers.Layer):
    """Residual in Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RRDB',
                 **kwargs):
        super(ResInResDenseBlock, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta
        self.rdb_1 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_2 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_3 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)

    def call(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return out * self.res_beta + x


def RRDB_Model(size, channels, cfg_net, gc=32, wd=0., name='RRDB_model'):
    """Residual-in-Residual Dense Block based Model """
    nf, nb = cfg_net['nf'], cfg_net['nb']
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, nf=nf, gc=gc, wd=wd)
    conv_f = functools.partial(Conv2D, kernel_size=3, padding='same',
                               bias_initializer='zeros',
                               kernel_regularizer=tf.keras.regularizers.l2(wd))
    rrdb_truck_f = tf.keras.Sequential(
        [rrdb_f(name="RRDB_{}".format(i)) for i in range(nb)],
        name='RRDB_trunk')

    # extraction
    x = inputs = Input([size, size, channels], name='input_image')
    fea = conv_f(filters=nf, name='conv_first')(x)
    fea_rrdb = rrdb_truck_f(fea)
    trunck = conv_f(filters=nf, name='conv_trunk')(fea_rrdb)
    fea = fea + trunck

    # upsampling
    size_fea_h = tf.shape(fea)[1] if size is None else size
    size_fea_w = tf.shape(fea)[2] if size is None else size
    fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2],
                                 method='nearest', name='upsample_nn_1')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_1')(fea_resize)
    fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4],
                                 method='nearest', name='upsample_nn_2')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_2')(fea_resize)
    fea = conv_f(filters=nf, activation=lrelu_f(), name='conv_hr')(fea)
    out = conv_f(filters=channels, name='conv_last')(fea)

    return Model(inputs, out, name=name)



    
def GEN(input_dim=8):
    mod=mod_inp = Input(shape=(input_dim,))
    mod=Reshape((1,1,8))(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(12)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(16)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(24)(mod)
    mod=c3c(32)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(32)(mod)
    mod=c3c(48)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(48)(mod)
    mod=c3c(54)(mod)
    mod=c3c(64)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(64)(mod)
    mod=c3c(72)(mod)
    mod=c3c(84)(mod)
    mod=c3c(94)(mod)
    mod=Conv2D(3,1,padding="same",activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def DIS(input_shape=(64,64,3,)):
    mod=mod_inp = Input(shape=input_shape)
    mod=c3c(16)(mod)
    mod=c3c(18)(mod)
    mod=c3c(24)(mod)
    mod=Conv2D(24,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(28)(mod)
    mod=c3c(32)(mod)
    mod=c3c(36)(mod)
    mod=Conv2D(36,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(42)(mod)
    mod=c3c(46)(mod)
    mod=c3c(48)(mod)
    mod=Conv2D(48,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(54)(mod)
    mod=c3c(64)(mod)
    mod=Conv2D(64,4,2,padding="same",activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=c3c(84)(mod)
    mod=c3c(84)(mod)
    mod=Conv2D(84,4,2,padding="same",activation="relu")(mod)
    mod=c3c(84)(mod)
    mod=LayerNormalization()(mod)
    mod=Dropout(0.05)(mod)
    mod=Flatten()(mod)
    mod=Dense(84,activation="relu")(mod)
    mod=Dense(64,activation="relu")(mod)
    mod=LayerNormalization()(mod)
    mod=Dropout(0.05)(mod)
    mod=Dense(32,activation="relu")(mod)
    mod=Dense(1,activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)
    
class gan():
    def __init__(self,trials=[],dim=8):
        self.dim=dim
        self.gen=GEN(input_dim=self.dim)
        self.dis=DIS()
    def pred(self,batch=4):
        return self.gen(np.random.rand(batch,self.dim).astype(np.float32))
    def train(self,data=[],epoch=1000,batch=16,predbatch=8):
        
        optimizer = keras.optimizers.SGD(0.003)
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
        labels=np.random.rand(data.shape[0],self.dim).astype(np.float32)
        
        
        for i in range(epoch):
            with tqdm(total=data.shape[0]) as pbar:
                while pbar.n+batch<pbar.total:
                    datum=data[pbar.n:pbar.n+batch];label=labels[pbar.n:pbar.n+batch];
                
                    with tf.GradientTape() as tape:
                        dis=self.dis(self.gen(label))
                        dis=keras.losses.binary_crossentropy(zeros,dis)
                        dis=tf.reduce_mean(dis) 
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        dis=self.dis(datum)
                        dis=keras.losses.binary_crossentropy(ones,dis)
                        dis=tf.reduce_mean(dis) 
                        grad=tape.gradient(dis,self.dis.trainable_variables)
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.dis.trainable_variables))
                        del tape
                    
                    with tf.GradientTape() as tape:
                        gen=self.gen(label)
                        dis_gen=self.dis(self.gen(label))
                        dis_gen=keras.losses.binary_crossentropy(ones,dis_gen)
                        dis_gen=tf.reduce_mean(dis_gen) 
                        ####mode seeking (tentative implement)###
                        roll_shift=random.randint(1,batch-1)
                        Lms=tf.reduce_mean(tf.abs(label-tf.roll(label,roll_shift,axis=0)),[1])
                        Lms/=tf.reduce_mean(tf.abs(gen-tf.roll(gen,roll_shift,axis=0)),[1,2,3])+np.full(batch,1e-5)
                        dis_gen+=tf.reduce_mean(Lms) 
                        ####mode seeking (tentative implement)###
                        grad=tape.gradient(dis_gen,self.gen.trainable_variables) 
                        grad,_ = tf.clip_by_global_norm(grad, 15)
                        optimizer.apply_gradients(zip(grad,self.gen.trainable_variables))
                        del tape
                    
                    pbar.update(batch)
                    
            print("\nke(fake,tru):",
                  tf.reduce_mean(keras.losses.binary_crossentropy (zeros,self.dis(self.gen(labels[:batch])))).numpy(),
                  tf.reduce_mean(keras.losses.binary_crossentropy (ones,self.dis(data[:batch]))).numpy())
            tf2img(self.pred(predbatch),os.path.join(args.outdir,"1"),epoch=i,ext=".png")
                    
            self.dis.save_weights(os.path.join(args.outdir,"disw.h5"))
            self.gen.save_weights(os.path.join(args.outdir,"genw.h5"))
            self.gen.save(os.path.join(args.outdir,"gen.h5"))
        

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train' ,help="train_data",default="./lfw")
parser.add_argument('-o', '--outdir' ,help="outdir",default="./output")
parser.add_argument('-b', '--batch' ,help="batch",default=32,type=int)
parser.add_argument('-p', '--predbatch' ,help="batch_size_of_prediction",default=8,type=int)
parser.add_argument('-e', '--epoch' ,help="epochs",default=50,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    tf_ini()
    mkdiring(args.outdir)
    img=img2np(ffzk(args.train),64)
    gans=gan()
    gans.train(img,epoch=args.epoch,batch=args.batch,predbatch=args.predbatch)
    
    
    
    
    
    
    