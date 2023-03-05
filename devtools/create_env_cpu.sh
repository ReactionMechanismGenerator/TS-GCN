#!/bin/bash -l

# This script does the following tasks:
# 	- creates the conda environment
# 	- installs PyTorch cpu version (i.e. CUDA None) https://pytorch.org/get-started/locally/
# 	- installs torch-geometric and the required dependencies in the environment: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


# Check if Micromamba is installed
if [ -x "$(command -v micromamba)" ]; then
    echo "Micromamba is installed."
    COMMAND_PKG=micromamba
# Check if Mamba is installed
elif [ -x "$(command -v mamba)" ]; then
    echo "Mamba is installed."
    COMMAND_PKG=mamba
# Check if Conda is installed
elif [ -x "$(command -v conda)" ]; then
    echo "Conda is installed."
    COMMAND_PKG=conda
else
    echo "Micromamba, Mamba, and Conda are not installed. Please download and install one of them - we strongly recommend Micromamba or Mamba."
    exit 1
fi

# Set up Conda/Micromamba environment
if [ "$COMMAND_PKG" == "micromamba" ]; then
    micromamba activate base
    BASE=$MAMBA_ROOT_PREFIX
    # shellcheck source=/dev/null
    source "$BASE/etc/profile.d/micromamba.sh"
else
    BASE=$(conda info --base)
    # shellcheck source=/dev/null
    source "$BASE/etc/profile.d/conda.sh"
fi
# Create conda environment
echo "Creating conda environment..."
$COMMAND_PKG create -n ts_gcn python=3.7 -y

# Activate the environment to install torch-geometric

echo "Activating ts_gcn environment"
if [ "$COMMAND_PKG" == "micromamba" ]; then
    micromamba activate ts_gcn
else
    conda activate ts_gcn
fi

# Install PyTorch
echo "Installing PyTorch..."
$COMMAND_PKG install pytorch==1.7.1 cpuonly -c pytorch -y

# Install torch-scatter and torch-sparse
echo "Installing torch-scatter and torch-sparse..."
pip install --verbose --no-cache-dir torch-scatter==2.0.7 torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.1+cpu.html

# Install torch-geometric
echo "Installing torch-geometric..."
pip install torch-geometric

# Install other dependencies
echo "Installing other dependencies..."
$COMMAND_PKG env update -f devtools/cpu_environment.yml
