# 機械学習を利用した、少数データでの画像高画質化手法の開発
### 超解像(8倍) (128×128 10000枚)
|DIV2K超解像|火星地表超解像|DIV2Kガウスノイズ除去|
|---|---|---|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/t570.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t529.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t236.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/p570.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p529.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p236.png)|

### DIV2Kでの超解像(4倍) 学習データ枚数
|HR|LR|10000枚|1000枚|100枚|
|---|---|---|---|---|
|PSNR平均|36.10|36.41|36.29|36.12|
|SSIM平均|0.759|0.775|0.771|0.759|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_HR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_LR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_10000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_1000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_100.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_HR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_LR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_10000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_1000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_100.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_HR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_LR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_10000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_1000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_100.png)|
## 使い方
### 使用前
`pip3 install -r requirements.txt`  
#### **DIV2K**と**Mars surface image (Curiosity rover) labeled data set**をDLして加工する  
`python3 download.py`  
#### 学習する  
`python3 inception2.py`  
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
Inception-Unet+SRCNN535(**inception2.py**)のデータを収集中。  
過学習について、凡そ128×128の画像100枚を教師とした場合、キュービック法とMSEで同じ精度となります。
