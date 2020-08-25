### はじめに
streamlitとkerasを使ってcrcr画像を認識する機械学習webアプリです。
- [video demo](https://twitter.com/irisuinwl/status/1298259707602202625)

### 環境情報

- Windows10: anaconda

### 設定

Anaconda Promptを開き、下記を実行

```
conda create -n keras python=3
conda activate keras
conda install tensorflow-gpu keras numpy pandas scikit-learn
pip install streamlit
```

### 利用ガイド

- `model`ディレクトリに学習モデルを格納する。
- 下記コマンドをAnaconda Promptで実行

```
streamlit run app.py
```
