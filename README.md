# TensorFlow2_Subclassing_MNIST
 
TensorFlow2.x系のSubclassing APIを用いてMNIST分類問題を解くサンプルコード

解説という程でもありませんが、[ブログ](https://qiita.com/ktokey/items/63d695d4c9f38c59b6d6)にて紹介しています。
 
# Requirement
 
* tensorflow 2.1 (or later)
* matplotlib
 
# Installation
 
Anaconda環境
 
```bash
conda install tensorflow-gpu=2.4.1
conda install matplotlib
```

pip環境

```bash
pip install tensorflow-gpu=2.4.1
pip install matplotlib
```
 
# Usage
 
下記のようにクローンして実行すると、自動的にファイル生成およびモデル学習を行うことができます。
 
```bash
git clone https://github.com/ktokey/TensorFlow2_Subclassing_MNIST.git
cd TensorFlow2_Subclassing_MNIST
python mnist_nn.py
python mnist_cnn.py
```