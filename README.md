# 深層学習を利用した、少数データでの画像高画質化手法の開発
### 高画質化 (128×128 10000枚)
|DIV2K超解像(8倍)|火星地表超解像(8倍)|DIV2Kガウスノイズ除去|
|---|---|---|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/t570.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t529.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/t236.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/p570.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p529.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/p236.png)|

## 背景 
 「データ数が十分に得られないリアルデータや観測データの分析」が求められる。  
例:火星や月面DEMの超解像,ヒートマップの高品質化など。  

このような限られたデータを**フロンティアデータ**と呼ぶ。  
そこで、**フロンティアデータ**の高品質化のための機械学習モデルの開発が必要である。  
## 目的
「限られたデータを元に画像データを高品質化する」機械学習手法を開発する。  
![](https://github.com/jSm449g4d/Research/blob/master/assets/selfteaching.png)  
## 課題
データ数が少ないと過学習という現象が発生する。  
過学習を極力抑えつつ、解像度の向上(超解像)や、ノイズ除去を行う仕組みを作る。  

### DIV2Kでの超解像(4倍) 学習データ枚数
|HR|LR|10000枚|1000枚|100枚|
|---|---|---|---|---|
|PSNR平均|24.07|24.70|24.44|24.10|
|SSIM平均|0.759|0.775|0.771|0.759|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_HR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_LR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_10000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_1000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/134_100.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_HR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_LR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_10000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_1000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/278_100.png)|
|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_HR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_LR.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_10000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_1000.png)|![](https://github.com/jSm449g4d/Research/blob/master/assets/349_100.png)|
## ディレクトリ構成
Research/  
┣assets/ (README.mdで使う画像置き場)  
┣wi2_paper/ (2020年のWI2研究会用の論文)**執筆中**  
┣dwonload_module/ (download.pyで使うデータセット加工用処理置き場)  置き場)  
┣util.py (機械学習で良く使う自作の関数置き場)  
┣requirememnts.txt (必要なパッケージ置き場)  
┣.gitignore (git pushでpushしたくないファイル一覧)  
┣cloudbuild.yaml (CaaSへのデプロイ指示書)  
┣LICENSE (MIT: ご自由にお使いください)  
┗README.md この文書  
## 使い方
### 使用前
`pip3 install -r requirements.txt`  
`sudo apt install graphviz`  
#### **DIV2K**と**Mars surface image (Curiosity rover) labeled data set**をDLして加工する  
`python3 download.py`  
#### 学習する  
`python3 inception2.py`  
## 提案するモデル(Inception-Unet+SRCNN535)
<details><summary>表示する</summary><div><img src="https://github.com/jSm449g4d/Research/blob/master/assets/model.png"/></div></details>

## 今後
新たな課題設定(ピンボケ画像の高画質化など)  
ヒートマップなどの「非画像の二次元データ」への応用  
機械学習モデルの改良  
## 進捗
Inception-Unet+SRCNN535(**inception2.py**)のデータを収集中。  
過学習について、凡そ128×128の画像100枚を教師とした場合、キュービック法とMSEで同じ精度となります。
