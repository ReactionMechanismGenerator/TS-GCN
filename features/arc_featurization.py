import torch
import torch_geometric as tg
from typing import TYPE_CHECKING, List, Type, Union

from arc.species.converter import check_xyz_dict, xyz_to_dmat
from arc.reaction import ARCReaction

from rmgpy.molecule.molecule import Atom, Molecule
from rmgpy.molecule.resonance import generate_optimal_aromatic_resonance_structures

from features.common import (ChiralType,
                             ATOM_FEATURES,
                             CHIRALTAG_PARITY,
                             ATOM_FDIM,
                             BOND_FDIM,
                             onek_encoding_unk)


def atom_features(atom: Type[Atom], aromatic: bool = False) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    Args:
        atom: An instance of rmgpy.molecule.molecule.Atom class.
        aromatic: Boolean for whether this atom is aromatic.
        functional_groups: A k-hot vector indicating the functional groups the atom belongs to.

    Returns:
        A list containing the atom features.
    """

    features = onek_encoding_unk(atom.symbol, ATOM_FEATURES['atomic_num']) + \
        [1 if aromatic else 0] + \
        onek_encoding_unk(len(atom.bonds), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.charge, ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(sum([k.symbol == 'H' for k in atom.bonds.keys()]), ATOM_FEATURES['num_Hs']) + \
        [atom.mass * 10]  # atom.mass [=] kg. Multiply by 10 (hectograms) to scale to similar range as other features

    return features


def get_aromatic_atom_indices(molecule: Type[Molecule]):
    """
    Determines the aromatic indices in an RMG-Py Molecule object

    Args:
        molecule: An RMG-Py Molecule object.

    Returns:
        aromatic_atom_indices: A list containing the indices corresponding to aromatic atoms.
                               If no atoms are aromatic, returns an empty list.
    """
    aromatic_atom_indices = list()
    resonance_structure = generate_optimal_aromatic_resonance_structures(molecule)
    # if there are aromatic resonance structures, identify the aromatic atoms
    if resonance_structure:
        resonance_structure = generate_optimal_aromatic_resonance_structures(molecule)[0]
        for atom in resonance_structure.atoms:
            for bond in atom.edges.values():
                if bond.is_benzene():
                    aromatic_atom_indices.append(resonance_structure.atoms.index(atom))
                    break
        return aromatic_atom_indices
    # else, resonance_structure is an empty list since the molecule contains no aromatic atoms, so return an empty list
    else:
        return aromatic_atom_indices


def featurization(reaction: Type[ARCReaction]):
    """
    Generates features of the reactant and product for one reaction as input for the network.

    Args:
        reaction: ARCReaction object for an isomerization reaction

    Returns:
        data: Torch Geometric Data object, storing the atom and bond features
        ts_xyz_dict: ARC xyz dictionary
    """
    # isomerization reactions have only 1 reactant and 1 product
    reactant = reaction.r_species[0]
    product = reaction.p_species[0]
    D_r = xyz_to_dmat(reactant.get_xyz())
    D_p = xyz_to_dmat(product.get_xyz())

    f_atoms = list()        # atom (node) features
    edge_index = list()     # list of tuples indicating presence of bonds
    f_bonds = list()        # bond (edge) features

    n_atoms = len(reactant.mol.atoms)  # number of atoms
    aromatic_atom_indices = get_aromatic_atom_indices(reactant.mol)

    for a1, atom in enumerate(reactant.mol.atoms):
        # atom features
        aromatic = True if a1 in aromatic_atom_indices else False
        f_atoms.append(atom_features(atom, aromatic))

        # edge features
        for a2 in range(a1 + 1, n_atoms):
            # create fully connected graph
            edge_index.extend([(a1, a2), (a2, a1)])

            # for now, naively include distance matrix for both reactant and product
            b1_feats = [D_r[a1][a2], D_p[a1][a2]]
            b2_feats = [D_r[a2][a1], D_p[a2][a1]]

            f_bonds.append(b1_feats)
            f_bonds.append(b2_feats)

    data = tg.data.Data()
    data.x = torch.tensor(f_atoms, dtype=torch.float)
    data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data.edge_attr = torch.tensor(f_bonds, dtype=torch.float)

    return data
