from rdkit import Chem
import numpy as np


def signed_volume(coords):

    v1 = coords[0] - coords[3]
    v2 = coords[1] - coords[3]
    v3 = coords[2] - coords[3]
    cp = np.cross(v2, v3)
    vol = np.dot(v1, cp)
    return np.sign(vol)


def check_volume_constraints(mols):
    r_mol = mols[0]
    ts_mol = mols[1]
    p_mol = mols[2]

    # find changed atoms
    r_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in r_mol.GetBonds()]
    p_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in p_mol.GetBonds()]
    broken_bonds = [bond for bond in r_bonds if bond not in p_bonds]
    formed_bonds = [bond for bond in p_bonds if bond not in r_bonds]
    changed_atoms = set.union(*broken_bonds + formed_bonds)

    try:
        # check signed volume constraints
        r_coords = r_mol.GetConformer().GetPositions()
        p_coords = p_mol.GetConformer().GetPositions()
        ts_coords = ts_mol.GetConformer().GetPositions()
        for a in r_mol.GetAtoms():
            if len(a.GetNeighbors()) == 4:
                if a.GetIdx() not in changed_atoms:
                    coord_ids = [n.GetIdx() for n in a.GetNeighbors()]
                    r_vol = signed_volume(r_coords[coord_ids])
                    p_vol = signed_volume(p_coords[coord_ids])
                    ts_vol = signed_volume(ts_coords[coord_ids])
                    assert r_vol == p_vol == ts_vol
        return True

    except AssertionError:
        return False


def clean_ts(mols):
    r_mol, ts_mol, p_mol = mols
    r_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in r_mol.GetBonds()]
    r_bonds = [tuple(sorted(b)) for b in r_bonds]
    p_bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in p_mol.GetBonds()]
    p_bonds = [tuple(sorted(b)) for b in p_bonds]
    common_bonds = list(set(r_bonds) & set(p_bonds))

    emol = Chem.EditableMol(ts_mol)
    for bond in ts_mol.GetBonds():
        bond_idxs = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if bond_idxs not in common_bonds:
            emol.RemoveBond(bond_idxs[0], bond_idxs[1])
            emol.AddBond(bond_idxs[0], bond_idxs[1])
    return r_mol, emol.GetMol(), p_mol
