from argparse import Namespace
import glob
import numpy as np
import os
from rdkit import Chem
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset, DataLoader
from typing import List, Tuple, Union

from features.common import (ATOM_FEATURES,
                             CHIRALTAG_PARITY,
                             ATOM_FDIM,
                             BOND_FDIM,
                             onek_encoding_unk,
                             )


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atomic_num']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features


def parity_features(atom: Chem.rdchem.Atom) -> int:
    """
    Returns the parity of an atom if it is a tetrahedral center.
    +1 if CW, -1 if CCW, and 0 if undefined/unknown
    :param atom: An RDKit atom.
    """
    return CHIRALTAG_PARITY[atom.GetChiralTag()]


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        # fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6))) # remove global cis/trans tags
        # fbond += [0, 0] # special cis/trans message edge type
    return fbond


def cistrans_bond_features(local_cistrans: str) -> List[Union[bool, int, float]]:
    fbond = [0 for _ in range(BOND_FDIM)]

    if local_cistrans == 'cis':
        fbond[-2] = 1
    elif local_cistrans == 'trans':
        fbond[-1] = 1
    else:
        raise ValueError('Invalid cis/trans specified')
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mols: str, args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.
        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.n_atoms = 0    # number of atoms
        self.n_bonds = 0    # number of bonds
        self.f_atoms = []   # mapping from atom index to atom features
        self.f_bonds = []   # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []       # mapping from atom index to incoming bond indices
        self.b2a = []       # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []    # mapping from bond index to the index of the reverse bond
        self.parity_atoms = []  # mapping from atom index to CW (+1), CCW (-1) or undefined tetra (0)
        self.edge_index = []    # list of tuples indicating presence of bonds
        self.y = []

        # extract reactant, ts, product
        r_mol, ts_mol, p_mol = mols

        # fake the number of "atoms" if we are collapsing substructures
        n_atoms = r_mol.GetNumAtoms()

        # topological and 3d distance matrices
        tD_r = Chem.GetDistanceMatrix(r_mol)
        tD_p = Chem.GetDistanceMatrix(p_mol)
        D_r = Chem.Get3DDistanceMatrix(r_mol)
        D_p = Chem.Get3DDistanceMatrix(p_mol)
        D_ts = Chem.Get3DDistanceMatrix(ts_mol)

        # temporary featurization
        for a1 in range(n_atoms):

            # Node features
            self.f_atoms.append(atom_features(r_mol.GetAtomWithIdx(a1)))

            # Edge features
            for a2 in range(a1 + 1, n_atoms):
                # fully connected graph
                self.edge_index.extend([(a1, a2), (a2, a1)])

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

                self.f_bonds.append(b1_feats)
                self.f_bonds.append(b2_feats)
                self.y.extend([D_ts[a1][a2], D_ts[a2][a1]])


class MolDataset(Dataset):

    def __init__(self, sdf_dir, args, mode='train'):
        super(MolDataset, self).__init__()

        if args.split_path:
            self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
            self.split = np.load(args.split_path, allow_pickle=True)[self.split_idx]
        else:
            self.split = list(range(len(smiles)))  # fix this

        self.sdf_dir = sdf_dir
        self.mols = self.get_mols()
        self.args = args

    def process_key(self, key):
        molgraph = MolGraph(self.mols[key], self.args)
        mol_data = self.molgraph2data(molgraph, key)
        return mol_data

    def molgraph2data(self, molgraph, key):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.mols = self.mols[key]
        data.y = torch.tensor(molgraph.y, dtype=torch.float)

        return data

    def get_mols(self):

        r_file = glob.glob(os.path.join(self.sdf_dir, '*_reactants.sdf'))[0]
        ts_file = glob.glob(os.path.join(self.sdf_dir, '*_ts.sdf'))[0]
        p_file = glob.glob(os.path.join(self.sdf_dir, '*_products.sdf'))[0]

        data = [Chem.SDMolSupplier(r_file, removeHs=False, sanitize=False),
                Chem.SDMolSupplier(ts_file, removeHs=False, sanitize=False),
                Chem.SDMolSupplier(p_file, removeHs=False, sanitize=False)]

        data = [(x, y, z) for (x, y, z) in zip(data[0], data[1], data[2]) if (x, y, z)]
        return [data[i] for i in self.split]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, key):
        return self.process_key(key)


def construct_loader(args, modes=('train', 'val')):

    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    for mode in modes:
        dataset = MolDataset(args.sdf_dir, args, mode)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True if mode == 'train' else False,
                            num_workers=args.num_workers,
                            pin_memory=True)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
