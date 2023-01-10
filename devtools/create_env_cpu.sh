# This script does the following tasks:
# 	- creates the conda environment
# 	- installs PyTorch cpu version (i.e. CUDA None) https://pytorch.org/get-started/locally/
# 	- installs torch-geometric and the required dependencies in the environment: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# Check for package manager
if command -v mamba >/dev/null 2>&1; then
    COMMAND_PKG=mamba
elif command -v conda >/dev/null 2>&1; then
    COMMAND_PKG=conda
else
    echo "Error: mamba or conda is not installed. Please download and install mamba or conda - we strongly recommend mamba"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
$COMMAND_PKG create -n ts_gcn python=3.7 -y

# Activate the environment to install torch-geometric
source ~/.bashrc
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
echo "Activating ts_gcn environment"
conda activate ts_gcn

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
$COMMAND_PKG env update -f cpu_environment.yml
