"""
This file reads in the sdf files for the reactant and product respectively to create RDKit molecules,
which are used to create features that are passed to the GCN to generate a TS guess. The TS guess is
written to an xyz file.

Example of use:
`python inference.py --r_sdf_path reactant.sdf --p_sdf_path product.sdf
"""
from argparse import ArgumentParser
import os
from rdkit import Chem
import torch
import torch_geometric as tg
from typing import TYPE_CHECKING, List, Type, Union
from torch_geometric.data import DataLoader
import yaml

from model.G2C import G2C
from model.common import ts_gen_path
from features.featurization import (atom_features,
                                    parity_features,
                                    bond_features,
                                    cistrans_bond_features)


def featurization(r_mol: Chem.rdchem.Mol,
                  p_mol: Chem.rdchem.Mol,
                  ):
    """
    Generates features of the reactant and product for one reaction as input for the network.

    Args:
        r_mol: RDKit molecule object for the reactant.
        p_mol: RDKit molecule object for the product.

    Returns:
        data: Torch Geometric Data object, storing the atom and bond features
    """

    # compute properties with rdkit (only works if dataset is clean)
    r_mol.UpdatePropertyCache()
    p_mol.UpdatePropertyCache()

    # fake the number of "atoms" if we are collapsing substructures
    n_atoms = r_mol.GetNumAtoms()

    # topological and 3d distance matrices
    tD_r = Chem.GetDistanceMatrix(r_mol)
    tD_p = Chem.GetDistanceMatrix(p_mol)
    D_r = Chem.Get3DDistanceMatrix(r_mol)
    D_p = Chem.Get3DDistanceMatrix(p_mol)

    f_atoms = list()        # atom (node) features
    edge_index = list()     # list of tuples indicating presence of bonds
    f_bonds = list()        # bond (edge) features

    for a1 in range(n_atoms):

        # Node features
        f_atoms.append(atom_features(r_mol.GetAtomWithIdx(a1)))

        # Edge features
        for a2 in range(a1 + 1, n_atoms):
            # fully connected graph
            edge_index.extend([(a1, a2), (a2, a1)])

            # for now, naively include both reac and prod
            b1_feats = [D_r[a1][a2], D_p[a1][a2]]
            b2_feats = [D_r[a2][a1], D_p[a2][a1]]

            # r_bond = r_mol.GetBondBetweenAtoms(a1, a2)
            # b1_feats.extend(bond_features(r_bond))
            # b2_feats.extend(bond_features(r_bond))
            #
            # p_bond = p_mol.GetBondBetweenAtoms(a1, a2)
            # b1_feats.extend(bond_features(p_bond))
            # b2_feats.extend(bond_features(p_bond))

            f_bonds.append(b1_feats)
            f_bonds.append(b2_feats)

    data = tg.data.Data()
    data.x = torch.tensor(f_atoms, dtype=torch.float)
    data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data.edge_attr = torch.tensor(f_bonds, dtype=torch.float)

    return data

def inference(r_mols: List[Chem.rdchem.Mol],
              p_mols: List[Chem.rdchem.Mol],
              ts_xyz_path: str = 'TS.xyz',
              ):
    """
    Loads in the best weights from hyperparameter optimization to predict a TS guess.
    The TS guess is written to an xyz file.

    Args:
        r_mol: List of RDKit molecule objects for the reactant/s present in the sdf file.
        p_mol: List of RDKit molecule objects for the product/s present in the sdf file.
        ts_xyz_path: String indicating the path to write the TS guess structure to.

    """
    # create torch data loader
    data_list = list()
    for r_mol, p_mol in zip(r_mols, p_mols):
        data = featurization(r_mol, p_mol)
    data_list.append(data)

    loader = DataLoader(data_list, batch_size=16)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define paths to model parameters and state dictionary
    yaml_file_name = os.path.join(ts_gen_path, 'best_model', 'model_parameters.yml')
    state_dict = os.path.join(ts_gen_path, 'best_model', 'best_model.pt')

    # create the network with the best architecture from hyperopt and load the corresponding best weights
    with open(yaml_file_name, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    model = G2C(**content).to(device)
    model.load_state_dict(torch.load(state_dict, map_location=device))
    model.eval()

    for i, data in enumerate(loader):
        data = data.to(device)
        out, mask = model(data)  # out is distance matrix. mask is matrix of 1s with 0s along diagonal

        symbols = [a.GetSymbol() for a in r_mol.GetAtoms()]
        for batch in data.coords:
            coords = batch.double().cpu().detach().numpy().tolist()

            # extract the coordinates and prepare a string to write to an xyz file
            xyz_list = list()
            for symbol, coord in zip(symbols, coords):
                row = '{0:4}'.format(symbol)
                row += '{0:14.8f}{1:14.8f}{2:14.8f}'.format(*coord)
                xyz_list.append(row)
            TS_xyz = '\n'.join(xyz_list)
            # add the number of atoms at the top of the xyz file
            TS_xyz = str(len(symbols)) + '\n' + '\n' + TS_xyz + '\n'
            with open(ts_xyz_path, 'w') as f:
                f.write(TS_xyz)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--r_sdf_path', type=str, default='reactant.sdf')
    parser.add_argument('--p_sdf_path', type=str, default='product.sdf')
    parser.add_argument('--ts_xyz_path', type=str, default='TS.xyz')
    args = parser.parse_args()

    # read in sdf files for reactant and product of the atom-mapped reaction
    r_mols = Chem.SDMolSupplier(args.r_sdf_path, removeHs=False, sanitize=True)
    p_mols = Chem.SDMolSupplier(args.p_sdf_path, removeHs=False, sanitize=True)
    inference(r_mols, p_mols, args.ts_xyz_path)
