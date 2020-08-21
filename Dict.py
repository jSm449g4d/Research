import numpy as np
import scipy as sci
from numpy.random import *
from PIL import Image
import os
import copy
import sys


def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

def prox_norm1(v, gamma, lam):
    return soft_thresh(v, lam * gamma)

def soft_thresh(b, lam):
    x_hat = np.zeros(b.shape[0])
    x_hat[b >= lam] = b[b >= lam] - lam
    x_hat[b <= -lam] = b[b <= -lam] + lam
    return x_hat

def proximal_gradient(grad_f, prox, gamma, objective, init_x, tol = 1e-9):
    x = init_x
    result = objective(x)
    while 1:
        x_new = prox(x - gamma * grad_f(x), gamma)
        result_new = objective(x_new)
        if (np.abs(result - result_new)/np.abs(result) < tol) == True :
            break;
        x = x_new
        result = result_new
    return x_new, result_new

def ISTA(W, y, lam):
    objective = lambda x:np.sum(np.power(y - np.dot(W, x), 2))/2.0 + lam * np.sum(np.abs(x)) #(1)
    grad_f = lambda x: np.dot(W.T, np.dot(W, x) - y)  #(2)
    (u, l, v) = np.linalg.svd(W)   #(3)
    gamma = 1/max(l.real*l.real)   #(4)
    prox = lambda v, gamma:prox_norm1(v, gamma, lam)  #(5)
    x_init = randn(W.shape[1])  #(6)
    (x_hat, result) = proximal_gradient(grad_f, prox, gamma, objective, x_init, 1e-5)  #(7)
    return x_hat, result

#おれおれ画像・ベクトル・行列変換関数群
def color2gray_img(img):
    return img.convert('L')

def im2vec(img):
    img_arr = np.array(img.convert('L'),'float')
    return img_arr.reshape(-1)

def gray_im2float_vec(gray_img):
    return im_scale2float((im2vec(gray_img)))

def color_im2float_vec(color_img):
    return gray_im2float_vec(color2gray_img(color_img))

def vec2image(vec, H, W):
    tmp_vec = copy.deepcopy(vec)
    tmp_vec = tmp_vec.reshape((H, W))
    return Image.fromarray(np.uint8(tmp_vec))

def float_vec2image(vec, H, W):
    return vec2image(float2im_scale(vec), H, W)

def arr2image(arr):
    arr = copy.deepcopy(arr)
    return Image.fromarray(np.uint8(arr))

def float_arr2image(arr):
    return arr2image(float2im_scale(arr))

def im_scale2float(vec):
    return vec.astype(np.float)

def float2im_scale(vec):
    vec[vec > 255.0] = 255.0
    vec[vec < 0.0] = 0.0
    return vec

#画像データからパッチを切り出す関数
#shift_numで切り出す際のずらし量を指定
def gray_im2patch_vec(im, patch_hight, patch_width, shift_num):
    patchs = np.zeros((patch_hight*patch_width,
                        int(1 + (im.size[0] - patch_hight)/shift_num + (1 if (im.size[0] - patch_hight) % shift_num  != 0 else 0)) *
                        int(1 + (im.size[1] - patch_width)/shift_num + (1 if (im.size[1] - patch_width) % shift_num  != 0 else 0))))
    im_arr = im_scale2float(np.array(im))
    k = 0
    for i in range(int(1 + (im.size[0] - patch_hight)/shift_num + (1 if (im.size[0] - patch_hight) % shift_num  != 0 else 0))):
        h_over_shift = min(0, im.size[0] - ((i* shift_num) + patch_hight)) #ずらしたことによって画像からはみ出る量(縦方向)
        for j in range(int(1 + (im.size[1] - patch_width)/shift_num + (1 if (im.size[1] - patch_width) % shift_num  != 0 else 0))):
            w_over_shift = min(0, im.size[1] - ((j* shift_num) + patch_width)) #ずらしたことによって画像からはみ出る量(横方向)
            #パッチの切り出し　画像からはみ出たら、そのはみ出た分を戻して切り出し
            patchs[:, k] = im_arr[(i* shift_num) + h_over_shift:((i* shift_num) + patch_hight) + h_over_shift,
                                    (j* shift_num) + w_over_shift:((j* shift_num) + patch_width) + w_over_shift].reshape(-1)
            k += 1
    return patchs


#式（5）を計算する関数
def dictionary_learning_with_mod(W, Y):
    D = np.dot(np.dot(Y, W.T), np.linalg.inv(np.dot(W, W.T)))
    result = np.linalg.norm(Y - np.dot(D, W)) ** 2
    return D, result

#辞書クラス
class dictionary_with_learning():
    def __init__(self, weight_learner, dictionary_learner, base_num, max_loop_num = 8):
        self.Dic = np.array([])
        self.weight_learner = weight_learner
        self.dictionary_learner = dictionary_learner
        self.base_num = base_num

    #学習データ行列Yから辞書を学習するメソッド
    def learn(self, Y, eps = 1e-8):
        #手順3:ランダム行列で辞書の初期化
        D = np.random.randn(Y.shape[0], self.base_num)

        result = -1
        loop_num = 0
        while 1:
            loop_num += 1
            W = np.zeros((self.base_num, Y.shape[1]))
            #手順4:Dを固定して係数を求める
            for i in range(Y.shape[1]):
                [W[:, i], tmp] = self.weight_learner(D, Y[:, i])

            #手順5:Wを固定してDを求める
            [D, new_result] = self.dictionary_learner(W, Y)

            if result < 0 or np.abs(new_result - result) > eps or loop_num < max_loop_num:
                result = new_result
            else: #手順4、5を収束するまで繰り返す。
                break;
        self.Dic = D
        return D

    def get_dic_array(self):
        return self.Dic



#画像パッチから学習される辞書クラス
class image_patch_dictionary():
    def __init__(self, dictionary):
        self.dictonary = dictionary
        self.patch_hight = 0
        self.patch_width = 0

    #学習画像の集合から辞書を学習するメソッド
    def learn_images(self, image_list, patch_hight, patch_width, eps=2e-2, shift_num=1):
        self.patch_hight = patch_hight
        self.patch_width = patch_width
        Y = np.empty((patch_hight * patch_width, 0))
        #手順1:画像パッチの切り出し
        for img in image_list:
            gray_img = color2gray_img(img)
            Ytmp = gray_im2patch_vec(gray_img, patch_hight, patch_width,shift_num) 
            Y = np.concatenate((Y, Ytmp), axis=1)

        #手順2:学習データ行列の標準化
        Y_ = (Y - np.mean(Y, axis=1, keepdims = True))/np.std(Y, axis = 1, keepdims=True)
        self.dictonary.learn(Y_, eps)

    def get_dic(self):
        return self.dictonary.get_dic_array()

    def get_patch_hight(self):
        return self.patch_hight

    def get_patch_width(self):
        return self.patch_width

#バックプロジェクションで使うダウンサンプラー行列を求める関数
#ダウンサンプラーはスパースかつサイズが大きい行列なのでscipy.sparseのスパース行列型として返す
def get_downsampler_arr(from_H, from_W, to_H, to_W):
    S = sci.sparse.lil_matrix((to_H * to_W, from_H * from_W))
    k = 0
    for i in range(0, from_H, from_H/to_H):
        for j in range(0, from_W, from_W/to_W):
            s = sci.sparse.lil_matrix((from_H, from_W))
            s[i : i + from_H/to_H, j:j + from_W/to_W] = (float(to_W)/float(from_W)) * (float(to_H)/float(from_H))
            S[k, :] = s.reshape((1, from_H*from_W))
            k += 1
    return  S.tocsr()

#ベクトルデータのダウンサンプリング関数
def vec_downsampling(vec, from_H, from_W, to_H, to_W):
    arr = vec.reshape((from_H, from_W))
    ret_arr = arr_downsampling(arr, from_H, from_W, to_H, to_W)
    return ret_arr.reshape(-1)

#行列データのダウンサンプリング関数
def arr_downsampling(arr, from_H, from_W, to_H, to_W):
    ret_arr = np.zeros((to_H, to_W))
    k = 0
    for i in range(0, from_H, from_H/to_H):
        for j in range(0, from_W, from_W/to_W):
            ret_arr[i/(from_H/to_H), j/(from_W/to_W)] = (np.average(arr[i : i + from_H/to_H, j:j + from_W/to_W]))
            k += 1
    return ret_arr

#画像パッチをつなぎ合わせて画像を再構成する関数
#shift_numがずらし量、パッチがオーバーラップしている個所は足し合わせて平均化
def patch_vecs2gray_im(patchs, im_hight, im_width, patch_hight, patch_width, shift_num):
    im_arr = np.zeros((im_hight, im_width))
    sum_count_arr = np.zeros((im_hight, im_width)) #各領域何回パッチを足したか表す行列
    i = 0
    j = 0
    w_over_shift = 0
    h_over_shift = 0
    for k in range(patchs.shape[1]):
        P = patchs[:, k].reshape((patch_hight, patch_width))
        im_arr[(i* shift_num) + h_over_shift:((i* shift_num) + patch_hight) + h_over_shift,
            (j* shift_num) + w_over_shift:((j* shift_num) + patch_width) + w_over_shift] += P #指定の領域にパッチの足しこみ
        sum_count_arr[(i* shift_num) + h_over_shift :(i* shift_num) + patch_hight + h_over_shift,
            (j* shift_num) + w_over_shift:(j* shift_num) + patch_width + w_over_shift] += 1 #当該領域のパッチを足しこんだ回数をカウントアップ
        if j < ((im_width - patch_width)/shift_num + (1 if (im_width - patch_width) % shift_num  != 0 else 0)):
            j += 1
            w_over_shift = min(0, im_width - ((j* shift_num) + patch_width)) #パッチを足しこむ領域が画像からはみ出た時、そのはみ出た分を戻すための変数（横方向）
        else:
            j = 0
            i += 1
            h_over_shift = min(0, im_hight - ((i* shift_num) + patch_hight)) #パッチを足しこむ領域が画像からはみ出た時、そのはみ出た分を戻すための変数（縦方向）
            w_over_shift = 0
    im_arr /= sum_count_arr
    return float_arr2image(im_arr), im_arr

#バックプロジェクションを行う関数
def back_projection(x0, y, L, c, mu, x_init = [], eps = 1e-7):
    if x_init == []:
        xt = sci.matrix(np.random.rand(x0.shape[0], x0.shape[1])) *255.0
    else:
        xt = x_init
    result = -1
    while 1:
        x_new = xt - mu * (L.T.tocsr()*(L * xt - y) + c* (xt - x0))
        result_new = sci.linalg.norm(L * x_new - y, 2) + c * sci.linalg.norm(x_new - x0, 2)
        if np.abs(result_new - result) < eps:
            break;
        xt = x_new
        result = result_new
    return x_new, result_new

#超解像計算クラス
class super_resolution:
    def __init__(self, dictionary, weight_learner, back_projection):
        self.Dh = dictionary.get_dic()
        self.patch_hight = dictionary.get_patch_hight()
        self.patch_width = dictionary.get_patch_width()
        self.weight_learner = weight_learner
        self.back_projection_method = back_projection

    def get_super_resolution(self, image, scaling, shift_num = 1):
        #手順1:画像パッチの切り出し
        image = color2gray_img(image)
        Y = gray_im2patch_vec(image, self.patch_hight/scaling, self.patch_width/scaling, shift_num)

        #手順2:辞書のダウンサンプリング
        Dl = np.zeros((self.Dh.shape[0]/(scaling**2), self.Dh.shape[1]))
        for i in range(self.Dh.shape[1]):
            Dl[:, i] = vec_downsampling(self.Dh[:, i], self.patch_hight, self.patch_width,
                                        self.patch_hight/scaling, self.patch_width/scaling)
       
        im_arr = np.zeros(image.size)
        i = 0
        j = 0
        Patchs = np.zeros(((self.patch_hight) * (self.patch_width), Y.shape[1]))
        W = np.zeros((self.Dh.shape[1], Y.shape[1]))
        for k in range(Y.shape[1]):
            #手順3:符号化係数の推定
            [w, result] = self.weight_learner(Dl, Y[:, k])

            #手順4:高解像画像パッチの生成
            Patchs[:, k] = np.dot(self.Dh, w)

        #手順5:パッチをつなぎ合わせて高解像画像を再構成
        [reconstructed_im, X0] = patch_vecs2gray_im(Patchs, image.size[0] * scaling , image.size[1] * scaling , self.patch_hight, self.patch_width, shift_num* scaling)
        L = get_downsampler_arr(image.size[0] * scaling, image.size[1] * scaling, image.size[0], image.size[1])

        #手順6:バックプロジェクションによる画像の補正
        x = np.array(self.back_projection_method(sci.matrix(X0.reshape(-1)).T, sci.matrix(np.array(image).reshape(-1)).T, L)[0])
        return float_arr2image(x.reshape((image.size[0] * scaling, image.size[1] * scaling)))


if __name__ == "__main__":
    lam = 5
    base_num = 128
    patch_hight = 8
    patch_width = 8
    
    dataset=ffzk(os.path.join("./", './div2k_srlearn/test_cubic4'))[3:]
    train_img_list = []
    for file in dataset:
        if os.path.isfile(file):
            try:
                train_img_list.append(Image.open(file))
                if len(train_img_list) >= 50:
                    break;
            except:
                pass

    weight_learning_method = lambda W, y:ISTA(W, y, lam) #近接勾配法を行う関数を使用
    dic_learning_method = lambda W, Y:dictionary_learning_with_mod(W, Y)#解析的に辞書Dを求める方法を使用
    im_dic = image_patch_dictionary(dictionary_with_learning(weight_learning_method, dic_learning_method, base_num))
    im_dic.learn_images(train_img_list, patch_hight, patch_width, shift_num = patch_hight)

    np.save('./dic_array.npy', im_dic.dictonary.Dic)
    
    
    test_img = ffzk(os.path.join("./", './div2k_srlearn/test_cubic4'))[0]
    scale = 3
    lam = 50
    shift_num = 1
    c = 0.8
    mu = 0.9
    weight_learning_method = lambda W, y:ISTA(W, y, lam)
    back_projection_method = lambda x0, y, L:back_projection(x0, y, L, c, mu)
    dic_learning_method = lambda W, Y:dictionary_learning_with_mod(W, Y)
    im_dic = image_patch_dictionary(dictionary_with_learning(weight_learning_method, dic_learning_method, base_num))
    im_dic.dictonary.Dic = np.load('./dic_array.npy')
    im_dic.patch_hight = 8
    im_dic.patch_width = 8
    SR = super_resolution(im_dic, weight_learning_method, back_projection_method)
    result_img = SR.get_super_resolution(test_img, scale, shift_num)
    result_img.save("./Lenna_SR.png")