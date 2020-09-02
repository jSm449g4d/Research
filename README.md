# 限られたデータを元に画像の高画質化を行う機械学習モデルの開発
|DIV2K超解像|火星地表超解像|DIV2Kガウスノイズ除去|
|---|---|---|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/t22.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t31.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t182.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/p22.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p31.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p182.png)|
## 使い方
### 使用前
`pip3 install -r requirements.txt`  
`python3 download.py`  
## 背景
現在ではビッグデータで学習した機械学習モデルを利用して問題解決を図ることが一般的である。  
例えばVGG16による画像分類、ESRGAN(VGG19)等の超解像が挙げられる。  

一方で「限られたデータセットだけで機械学習すること」が必要なケースも存在する。  
1. エビデンスが求められる場合(月面や火星の高画質化等)  
2. 類似するデータがビッグデータに存在しない場合(月面DEMの超解像等)  

そこで、「限られたデータを元に画像の高画質化する」機械学習モデルの開発を行う。  
![](https://github.com/jSm449g4d/Research/blob/master/assets/selfteaching.png)
