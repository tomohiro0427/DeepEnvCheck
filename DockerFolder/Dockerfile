# ベースイメージ
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 非対話型でタイムゾーン設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 基本パッケージ更新＆開発用ライブラリインストール
RUN apt update && \
    apt install -y --no-install-recommends \
        git vim nano \
        python3 python3-pip python3-venv python3-dev \
        build-essential gcc g++ make \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        wget curl llvm libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev \
        libjpeg-dev libpng-dev \
        libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
        unzip zip tar \
        htop tmux screen && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Python / pip バージョン確認（ログ用）
RUN python3 --version
RUN pip3 --version

# Anacondaインストール
ARG conda_ver="2025.06-1"
RUN wget https://repo.anaconda.com/archive/Anaconda3-${conda_ver}-Linux-x86_64.sh && \
    bash Anaconda3-${conda_ver}-Linux-x86_64.sh -b && \
    rm Anaconda3-${conda_ver}-Linux-x86_64.sh

# Anaconda パス設定
ENV PATH=$PATH:/root/anaconda3/bin

# conda初期化
RUN conda init bash
# Anaconda チャネル TOS 承諾
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# conda アップデート
RUN conda update -n base -c defaults conda -y && \
    conda clean -afy


# 仮想環境作成
RUN conda create -n pytorch28-cuda128 python=3.10 -y
RUN echo "conda activate pytorch28-cuda128" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=pytorch28-cuda128
ENV PATH=/root/conda/envs/pytorch28-cuda128/bin:$PATH
# 仮想環境でライブラリインストール
SHELL ["conda", "run", "-n", "pytorch28-cuda128", "/bin/bash", "-c"]

# PyTorch + torchvision + torchaudio（CUDA対応）
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# requirements.txt をコピーしてインストール
COPY requirements.txt /workspace/requirements.txt
# その他ライブラリは requirements.txt から
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# デフォルトで仮想環境を有効化
SHELL ["/bin/bash", "--login", "-c"]

CMD ["bash"]
