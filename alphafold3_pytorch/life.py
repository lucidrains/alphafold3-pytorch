import gemmi
import os
import torch

import rdkit.Geometry.rdGeometry as rdGeometry

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol
from typing import Literal

from alphafold3_pytorch.common import amino_acid_constants, dna_constants, rna_constants
from alphafold3_pytorch.tensor_typing import Int, typecheck

def is_unique(arr):
    return len(arr) == len({*arr})

# human amino acids
# reordered so [N][...][C][OH] - [OH] is removed for all peptides except last

# NOTE: template SMILES were derived via `print([Chem.MolToSmiles(resname_to_mol[resname], canonical=False) for resname in resname_to_mol])`
# to guarantee the order (and quantity) of atoms in the SMILES string perfectly matches the atoms in the residue template structure

HUMAN_AMINO_ACIDS = dict(
    A = dict(
        smile = 'CC(C(=O)O)N',
        # template_smile = 'NC(C=O)C',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    R = dict(
        smile = 'C(CC(C(=O)O)N)CN=C(N)N',
        # template_smile = 'NC(C=O)CCCNC(N)=N',
        first_atom_idx = 6,
        last_atom_idx = 3,
        hydroxyl_idx = 5,
        distogram_atom_idx = 2
    ),
    N = dict(
        smile = 'C(C(C(=O)O)N)C(=O)N',
        # template_smile = 'NC(C=O)CC(=O)N',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    D = dict(
        smile = 'C(C(C(=O)O)N)C(=O)O',
        # template_smile = 'NC(C=O)CC(=O)O',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 8,
        distogram_atom_idx = 1
    ),
    C = dict(
        smile = 'C(C(C(=O)O)N)S',
        # template_smile = 'NC(C=O)CS',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    Q = dict(
        smile = 'C(CC(=O)N)C(C(=O)O)N',
        # template_smile = 'NC(C=O)CCC(=O)N',
        first_atom_idx = 9,
        last_atom_idx = 6,
        hydroxyl_idx = 8,
        distogram_atom_idx = 5
    ),
    E = dict(
        smile = 'C(CC(=O)O)C(C(=O)O)N',
        # template_smile = 'NC(C=O)CCC(=O)O',
        first_atom_idx = 9,
        last_atom_idx = 6,
        hydroxyl_idx = 8,
        distogram_atom_idx = 5
    ),
    G = dict(
        smile = 'C(C(=O)O)N',
        # template_smile = 'NCC=O',
        first_atom_idx = 4,
        last_atom_idx = 1,
        hydroxyl_idx = 3,
        distogram_atom_idx = 0
    ),
    H = dict(
        smile = 'C1=C(NC=N1)CC(C(=O)O)N',
        # template_smile = 'NC(C=O)CC1=CNC=N1',
        first_atom_idx = 10,
        last_atom_idx = 7,
        hydroxyl_idx = 9,
        distogram_atom_idx = 0
    ),
    I = dict(
        smile = 'CCC(C)C(C(=O)O)N',
        # template_smile = 'NC(C=O)C(CC)C',
        first_atom_idx = 8,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 0
    ),
    L = dict(
        smile = 'CC(C)CC(C(=O)O)N',
        # template_smile = 'NC(C=O)CC(C)C',
        first_atom_idx = 8,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 4
    ),
    K = dict(
        smile = 'C(CCN)CC(C(=O)O)N',
        # template_smile = 'NC(C=O)CCCCN',
        first_atom_idx = 9,
        last_atom_idx = 6,
        hydroxyl_idx = 8,
        distogram_atom_idx = 5
    ),
    M = dict(
        smile = 'CSCCC(C(=O)O)N',
        # template_smile = 'NC(C=O)CCSC',
        first_atom_idx = 8,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 4
    ),
    F = dict(
        smile = 'C1=CC=C(C=C1)CC(C(=O)O)N',
        # template_smile = 'NC(C=O)CC1=CC=CC=C1',
        first_atom_idx = 11,
        last_atom_idx = 8,
        hydroxyl_idx = 10,
        distogram_atom_idx = 7
    ),
    P = dict(
        smile = 'C1CC(NC1)C(=O)O',
        # template_smile = 'N1C(C=O)CCC1',
        first_atom_idx = 3,
        last_atom_idx = 5,
        hydroxyl_idx = 7,
        distogram_atom_idx = 2
    ),
    S = dict(
        smile = 'C(C(C(=O)O)N)O',
        # template_smile = 'NC(C=O)CO',
        first_atom_idx = 5,
        last_atom_idx = 2,
        hydroxyl_idx = 4,
        distogram_atom_idx = 1
    ),
    T = dict(
        smile = 'CC(C(C(=O)O)N)O',
        # template_smile = 'NC(C=O)C(O)C',
        first_atom_idx = 6,
        last_atom_idx = 3,
        hydroxyl_idx = 5,
        distogram_atom_idx = 2
    ),
    W = dict(
        smile = 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
        # template_smile = 'NC(C=O)CC1=CNC2=C1C=CC=C2',
        first_atom_idx = 14,
        last_atom_idx = 11,
        hydroxyl_idx = 13,
        distogram_atom_idx = 10
    ),
    Y = dict(
        smile = 'C1=CC(=CC=C1CC(C(=O)O)N)O',
        # template_smile = 'NC(C=O)CC1=CC=C(O)C=C1',
        first_atom_idx = 11,
        last_atom_idx = 8,
        hydroxyl_idx = 10,
        distogram_atom_idx = 7
    ),
    V = dict(
        smile = 'CC(C)C(C(=O)O)N',
        # template_smile = 'NC(C=O)C(C)C',
        first_atom_idx = 7,
        last_atom_idx = 4,
        hydroxyl_idx = 6,
        distogram_atom_idx = 3
    )
)

# nucleotides
# reordered from 5' to 3', so [O][P][...][C(3')][OH] - hydroxyl group removed when chaining into a nucleic acid chain

DNA_NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1C(C(OC1N2C=NC3=C(N=CN=C32)N)COP(=O)(O)O)O',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O',
        complement = 'T',
        first_atom_idx = 20,
        last_atom_idx = 1,
        hydroxyl_idx = 21,
        distogram_atom_idx = 4
    ),
    C = dict(
        smile = 'C1C(C(OC1N2C=CC(=NC2=O)N)COP(=O)(O)O)O',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1O',
        complement = 'G',
        first_atom_idx = 17,
        last_atom_idx = 1,
        hydroxyl_idx = 19,
        distogram_atom_idx = 4
    ),
    G = dict(
        smile = 'C1C(C(OC1N2C=NC3=C2N=C(NC3=O)N)COP(=O)(O)O)O',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1O',
        complement = 'C',
        first_atom_idx = 21,
        last_atom_idx = 1,
        hydroxyl_idx = 22,
        distogram_atom_idx = 4
    ),
    T = dict(
        smile = 'CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)O)O',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1O',
        complement = 'A',
        first_atom_idx = 19,
        last_atom_idx = 11,
        hydroxyl_idx = 20,
        distogram_atom_idx = 9
    )
)

RNA_NUCLEOTIDES = dict(
    A = dict(
        smile = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O)N',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O',
        complement = 'U',
        first_atom_idx = 19,
        last_atom_idx = 11,
        hydroxyl_idx = 20,
        distogram_atom_idx = 9
    ),
    C = dict(
        smile = 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)COP(=O)([O-])[O-])O)O',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1O',
        complement = 'G',
        first_atom_idx = 17,
        last_atom_idx = 10,
        hydroxyl_idx = 19,
        distogram_atom_idx = 8
    ),
    G = dict(
        smile = 'C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)O)O)O)N=C(NC2=O)N',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)C(O)C1O',
        complement = 'C',
        first_atom_idx = 14,
        last_atom_idx = 7,
        hydroxyl_idx = 16,
        distogram_atom_idx = 5
    ),
    U = dict(
        smile = 'C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)(O)O)O)O',
        # template_smile = 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1O',
        complement = 'A',
        first_atom_idx = 18,
        last_atom_idx = 10,
        hydroxyl_idx = 19,
        distogram_atom_idx = 8
    )
)

# complements in tensor form, following the ordering ACG(T|U)N

NUCLEIC_ACID_COMPLEMENT_TENSOR = torch.tensor([3, 2, 1, 0, 4], dtype = torch.long)

# some functions for nucleic acids

@typecheck
def reverse_complement(
    seq: str,
    nucleic_acid_type: Literal['dna', 'rna'] = 'dna'
):
    if nucleic_acid_type == 'dna':
        nucleic_acid_entries = DNA_NUCLEOTIDES
    elif nucleic_acid_type == 'rna':
        nucleic_acid_entries = RNA_NUCLEOTIDES

    assert all([nuc in nucleic_acid_entries for nuc in seq]), 'unknown nucleotide for given nucleic acid type'

    complement = [nucleic_acid_entries[nuc]['complement'] for nuc in seq]
    return ''.join(complement[::-1])

@typecheck
def reverse_complement_tensor(t: Int[' n']):
    complement = NUCLEIC_ACID_COMPLEMENT_TENSOR[t]
    reverse_complement = complement.flip(dims = (-1,))
    return reverse_complement

# metal ions

METALS = dict(
    Mg = dict(
        smile = '[Mg]'
    ),
    Mn = dict(
        smile = '[Mn]'
    ),
    Fe = dict(
        smile = '[Fe]'
    ),
    Co = dict(
        smile = '[Co]'
    ),
    Ni = dict(
        smile = '[Ni]'
    ),
    Cu = dict(
        smile = '[Cu]'
    ),
    Zn = dict(
        smile = '[Zn]'
    ),
    Na = dict(
        smile = '[Na]'
    ),
    Cl = dict(
        smile = '[Cl]'
    )
)

# miscellaneous

MISC = dict(
    Phospholipid = dict(
        smile = 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)O)OC(=O)CCCCCCCC1CC1CCCCCC'
    )
)

# atoms - for atom embeddings

ATOMS = [
    'C',
    'O',
    'N',
    'S',
    'P',
    *METALS
]

assert is_unique(ATOMS)

# bonds for atom bond embeddings

ATOM_BONDS = [
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC'
]

assert is_unique(ATOM_BONDS)

# some rdkit helper function

@typecheck
def generate_conformation(mol: Mol) -> Mol:
    mol = Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mol, numConfs = 1)
    mol = Chem.RemoveHs(mol)
    return mol

@typecheck
def mol_from_smile(smile: str) -> Mol:
    mol = Chem.MolFromSmiles(smile)
    return generate_conformation(mol)

@typecheck
def remove_atom_from_mol(mol: Mol, atom_idx: int) -> Mol:
    edit_mol = Chem.EditableMol(mol)
    edit_mol.RemoveAtom(atom_idx)
    return mol

@typecheck
def mol_from_template_mmcif_file(
    mmcif_filepath: str, remove_hs: bool = True, remove_hydroxyl_oxygen: bool = True
) -> Chem.Mol:
    """
    Load an RDKit molecule from a template mmCIF file.

    Note that template atom positions are by default installed for each atom.
    This means users of this function should override these default atom
    positions as needed.

    :param mmcif_filepath: The path to a residue/ligand template mmCIF file.
    :param remove_hs: Whether to remove hydrogens from the template molecule.
    :param remove_hydroxyl_oxygen: Whether to remove the hydroxyl oxygen atom in each residue.
    :return: A corresponding template RDKit molecule.
    """
    # Parse the mmCIF file using Gemmi
    doc = gemmi.cif.read(mmcif_filepath)
    block = doc.sole_block()

    # Extract atoms and bonds
    atom_table = block.find(
        "_chem_comp_atom.",
        ["atom_id", "type_symbol", "model_Cartn_x", "model_Cartn_y", "model_Cartn_z"],
    )
    bond_table = block.find(
        "_chem_comp_bond.",
        ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag", "pdbx_stereo_config"],
    )

    # Create an empty `rdkit.Chem.RWMol` object
    mol = Chem.RWMol()

    # Dictionary to map atom ids to RDKit atom indices
    atom_id_to_idx = {}

    # Add atoms to the molecule
    for row in atom_table:
        element = row["type_symbol"]
        atom_id = row["atom_id"]
        if remove_hs and element == "H":
            continue
        elif remove_hydroxyl_oxygen and atom_id == "OXT":
            # NOTE: Hydroxyl oxygens are not present in the PDB's nucleotide residue templates
            continue
        rd_atom = Chem.Atom(element)
        idx = mol.AddAtom(rd_atom)
        atom_id_to_idx[atom_id] = idx

    # Create a conformer to store atom positions
    conf = Chem.Conformer(mol.GetNumAtoms())

    # Set atom coordinates
    for row in atom_table:
        atom_id = row["atom_id"]
        if atom_id not in atom_id_to_idx:
            continue
        idx = atom_id_to_idx[atom_id]
        x = float(row["model_Cartn_x"])
        y = float(row["model_Cartn_y"])
        z = float(row["model_Cartn_z"])
        conf.SetAtomPosition(idx, rdGeometry.Point3D(x, y, z))

    # Add conformer to the molecule
    mol.AddConformer(conf)

    # Add bonds to the molecule
    bond_order = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "AROM": Chem.BondType.AROMATIC,
    }

    for row in bond_table:
        atom_id1 = row["atom_id_1"]
        atom_id2 = row["atom_id_2"]
        if atom_id1 not in atom_id_to_idx or atom_id2 not in atom_id_to_idx:
            continue
        order = row["value_order"]
        aromatic_flag = row["pdbx_aromatic_flag"]
        stereo_config = row["pdbx_stereo_config"]

        idx1 = atom_id_to_idx[atom_id1]
        idx2 = atom_id_to_idx[atom_id2]

        mol.AddBond(idx1, idx2, bond_order[order])

        if aromatic_flag == "Y":
            mol.GetBondBetweenAtoms(idx1, idx2).SetIsAromatic(True)

        # Handle stereochemistry
        if stereo_config == "N":
            continue
        elif stereo_config == "E":
            mol.GetBondBetweenAtoms(idx1, idx2).SetStereo(Chem.BondStereo.STEREOE)
        elif stereo_config == "Z":
            mol.GetBondBetweenAtoms(idx1, idx2).SetStereo(Chem.BondStereo.STEREOZ)

    # Convert `RWMol` to `Mol`
    mol = mol.GetMol()

    return mol

# pre-load all PDB amino acid and nucleotide residue templates as `rdkit.Chem` molecules

resname_to_mol = {}
for resnames in (amino_acid_constants.resnames, rna_constants.resnames, dna_constants.resnames):
    for resname in resnames:
        template_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chemical", f"{resname}.cif"
        )
        if os.path.exists(template_filepath):
            resname_to_mol[resname] = mol_from_template_mmcif_file(template_filepath)
        else:
            print(
                f"WARNING: Template residue file {template_filepath} not found, skipping pre-loading of this template..."
            )

# initialize rdkit.Chem with canonical SMILES

CHAINABLE_BIOMOLECULES = [
    *HUMAN_AMINO_ACIDS.values(),
    *DNA_NUCLEOTIDES.values(),
    *RNA_NUCLEOTIDES.values()
]

METALS_AND_MISC = [
    *METALS.values(),
    *MISC.values(),
]

for entry in [*CHAINABLE_BIOMOLECULES, *METALS_AND_MISC]:
    mol = mol_from_smile(entry['smile'])
    entry['rdchem_mol'] = mol

# reorder all the chainable biomolecules
# to simplify chaining them up and specifying the peptide or phosphodiesterase bonds

for entry in CHAINABLE_BIOMOLECULES:
    mol = entry['rdchem_mol']

    atom_order = torch.arange(mol.GetNumAtoms())

    atom_order[entry['first_atom_idx']] = -1
    atom_order[entry['last_atom_idx']] = 1e4
    atom_order[entry['hydroxyl_idx']] = 1e4 + 1

    atom_reorder = atom_order.argsort()

    mol = Chem.RenumberAtoms(mol, atom_reorder.tolist())

    entry.update(
        atom_reorder_indices = atom_reorder,
        rdchem_mol = mol
    )
