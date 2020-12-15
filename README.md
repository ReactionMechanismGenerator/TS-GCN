# ts_gen_v2


## Installation
Run `make conda_env` to create the conda environment. 
The script will request the user to enter one of the supported CUDA versions listed here: https://pytorch.org/get-started/locally/.
The script uses this CUDA version to install PyTorch and PyTorch Geometric. Alternatively, the user could manually follow the steps to install PyTorch Geometric here: https://github.com/rusty1s/pytorch_geometric/blob/master/.travis.yml.

## Usage
To run the network, provide an sdf file for the reactant and product to `inference.py`, which will use the optimal weights to provide a TS guess that is written to an xyz file; this geometry can be further optimized with a QM method.

`python inference.py --r_sdf_path examples/rxn0/reactant.sdf --p_sdf_path examples/rxn0/product.sdf`
