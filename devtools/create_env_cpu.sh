# This script does the following tasks:
# 	- creates the conda
# 	- installs torch torch-geometric in the environment


# get OS type
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=MacOS;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "Running ${machine}..."

# "cpuonly" works for Linux and Windows
CUDA="cpuonly"
# Mac does not use "cpuonly"
if [ $machine == "Mac" ]
then
    CUDA=" "
fi  
CUDA_VERSION="cpu"

echo "Creating conda environment..."
echo "Running: conda env create -f environment.yml"
conda env create -f travis_environment.yml

# activate the environment to install torch-geometric
source activate ts_gen_v2

echo "Installing torch-geometric..."
echo "Using CUDA version: $CUDA_VERSION"
# get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Using PyTorch version: $TORCH_VERSION"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric
