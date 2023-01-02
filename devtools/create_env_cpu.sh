# This script does the following tasks:
# 	- creates the conda environment
# 	- installs PyTorch cpu version (i.e. CUDA None) https://pytorch.org/get-started/locally/
# 	- installs torch-geometric and the required dependencies in the environment: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


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

#Checking for package manager
if [ -x "$(command -v mamba)" ]; then
	echo "mamba is installed."
	COMMAND_PKG=mamba
elif [ -x "$(command -v conda)" ]; then
	echo "conda is installed."
	COMMAND_PKG=conda
else
    echo "mamba and conda are not installed. Please download and install mamba or conda - we strongly recommend mamba"
fi

echo "Creating conda environment..."
echo "Running: $COMMAND_PKG create -n ts_gcn python=3.7.11 -y"
$COMMAND_PKG create -n ts_gcn python=3.7 -y

# activate the environment to install torch-geometric
echo "Checking which python"
which python
export PATH=$CONDA_PREFIX/bin:$PATH
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
echo "Activating ts_gcn environment"
conda activate ts_gcn
echo "Checking which python"
which python


echo "Running: $COMMAND_PKG install pytorch==1.7.1 cpuonly -c pytorch -y"
$COMMAND_PKG install pytorch==1.7.1 cpuonly -c pytorch -y

echo "Installing torch-scatter and torch-sparse..."
pip install --verbose --no-cache-dir torch-scatter==2.0.7 torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.1+cpu.html

echo "Installing torch-geometric..."
pip install torch-geometric

echo "Installing other dependencies..."
$COMMAND_PKG env update -f cpu_environment.yml

