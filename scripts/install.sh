# git clone https://github.com/haotian-liu/LLaVA.git

cd LLaVA

pip install -e .

cd ..

pip install -r requirements.txt

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install protobuf

conda install nvidia/label/cuda-11.8.0::cuda-toolkit

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow-gpu==2.9.2

python -m pip install pycocotools

# pip install hfai[full] --extra-index-url https://pypi.hfai.high-flyer.cn/simple --trusted-host pypi.hfai.high-flyer.cn