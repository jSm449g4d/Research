# 機械学習を利用した、少数データでの画像高画質化手法の開発
|DIV2K超解像|火星地表超解像|DIV2Kガウスノイズ除去|
|---|---|---|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/t22.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t31.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t182.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/p22.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p31.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p182.png)|
## 使い方
### 使用前
`pip3 install -r requirements.txt`  
**DIV2K**と**Mars surface image (Curiosity rover) labeled data set**をDLして加工する  
`python3 download.py`  
## 背景
現在ではビッグデータで学習した機械学習モデルを再利用して、問題解決を図ることが一般的である。  
例えばVGG16による画像分類、ESRGAN(VGG19)等の超解像が挙げられる。  

一方で「限られたデータセットだけを手掛かりに、機械学習によるデータ高品質化」が求められるケースも存在する。  
1. エビデンスが求められる場合(月面や火星の高画質化等)  
2. 類似するデータは、ビッグデータではない場合(月面DEMの超解像等)  

このような限られたデータを**フロンティアデータ**と呼ぶ。  
そこで、**フロンティアデータ**の高品質化のための機械学習モデルの開発が必要である。  
## 目的
「限られた画像データを元に画像の高画質化する機械学習モデル」を開発する。
![](https://github.com/jSm449g4d/Research/blob/master/assets/selfteaching.png)
## 進捗
U-Netとほとんど同じスコアを出して、少ないデータセット(128×128のサブ画像1000枚程度)の学習可能なモデル**U-Inception**(uinception_2.py)を開発、現在詳細データを取っています。
