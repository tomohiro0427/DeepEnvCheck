
# DeepEnvCheck: PyTorch 環境チェック & 画像分類サンプル

このプロジェクトは、PyTorch と関連ライブラリが正しく動作するかを確認するための **簡単な画像分類サンプル**です。  
CIFAR-10 を使ったサンプル推論


## 実行環境構築
### 1. ローカルでの実行（Anaconda 推奨）
　1. Python 3.10 の仮想環境を作成：

```bash
conda create -n deepenvcheck python=3.10 -y
conda activate deepenvcheck
````
　2.　必要なライブラリをインストール：

```bash
# PyTorch + torchvision + torchaudio（CUDA対応）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Hugging Face Transformers, Datasets, Pillow
pip install transformers datasets pillow
```

---

### 2. Docker コンテナでの実行

Docker を使えば、環境構築の手間を減らして簡単に動かせます。

### 2-1. コンテナ起動

```bash
docker compose run --rm cuda
```
* `--rm`はコンテナ内で`exit`した場合、docker coposeもdownされる
* `cuda`の部分はdocker-compose.yamlに依存する

#### Python のインストール（コンテナ内）

```bash
# 対話型のタイムゾーン設定をスキップ
export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Tokyo

# パッケージ更新＆Pythonインストール
apt update
apt install -y python3 python3-pip python3-venv python3-dev build-essential

# バージョン確認
python3 --version
pip3 --version
```


### 2-2. 仮想環境の作成と管理（VENV）

既存の仮想環境を使う場合：

```bash
source venv/bin/activate
```

新しく作る場合：

```bash
# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# pip を最新化
pip install --upgrade pip

# 必要なライブラリをインストール
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets pillow
```

---

## 4. プログラムの実行
サンプルスクリプトを実行：

```bash
python main.py
```

* 推論結果（クラス ID とラベル名）がコンソールに出力されます。

---

## 5. 注意点・補足

* CUDA 対応 GPU がある場合は、自動で GPU 上で推論されます。
* CPU のみの場合でも動作可能ですが、推論は少し遅くなります。
* 本プロジェクトは **PyTorch 環境チェック用** が主目的で、学習は行いません。

