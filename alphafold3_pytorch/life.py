import os
from beartype.typing import Literal

import gemmi
import rdkit.Geometry.rdGeometry as rdGeometry
import torch
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.tensor_typing import Int, typecheck


def exists(v):
    return v is not None

def is_unique(arr):
    """Check if all elements in an array are unique."""
    return len(arr) == len({*arr})


# human amino acids

# NOTE: template SMILES were derived via `print([Chem.MolToSmiles(resname_to_mol[resname], canonical=False) for resname in resname_to_mol])`
# to guarantee the order (and quantity) of atoms in the SMILES string perfectly matches the atoms in the residue template structure

HUMAN_AMINO_ACIDS = dict(
    A=dict(
        resname="ALA",
        smile="NC(C=O)C",
        first_atom_idx=0,
        last_atom_idx=4,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    R=dict(
        resname="ARG",
        smile="NC(C=O)CCCNC(N)=N",
        first_atom_idx=0,
        last_atom_idx=10,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    N=dict(
        resname="ASN",
        smile="NC(C=O)CC(=O)N",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    D=dict(
        resname="ASP",
        smile="NC(C=O)CC(=O)O",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    C=dict(
        resname="CYS",
        smile="NC(C=O)CS",
        first_atom_idx=0,
        last_atom_idx=5,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    Q=dict(
        resname="GLN",
        smile="NC(C=O)CCC(=O)N",
        first_atom_idx=0,
        last_atom_idx=8,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    E=dict(
        resname="GLU",
        smile="NC(C=O)CCC(=O)O",
        first_atom_idx=0,
        last_atom_idx=8,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    G=dict(
        resname="GLY",
        smile="NCC=O",
        first_atom_idx=0,
        last_atom_idx=3,
        distogram_atom_idx=1,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    H=dict(
        resname="HIS",
        smile="NC(C=O)CC1=CNC=N1",
        first_atom_idx=0,
        last_atom_idx=9,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    I=dict(
        resname="ILE",
        smile="NC(C=O)C(CC)C",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    L=dict(
        resname="LEU",
        smile="NC(C=O)CC(C)C",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    K=dict(
        resname="LYS",
        smile="NC(C=O)CCCCN",
        first_atom_idx=0,
        last_atom_idx=8,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    M=dict(
        resname="MET",
        smile="NC(C=O)CCSC",
        first_atom_idx=0,
        last_atom_idx=7,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    F=dict(
        resname="PHE",
        smile="NC(C=O)CC1=CC=CC=C1",
        first_atom_idx=0,
        last_atom_idx=10,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    P=dict(
        resname="PRO",
        smile="N1C(C=O)CCC1",
        first_atom_idx=0,
        last_atom_idx=6,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    S=dict(
        resname="SER",
        smile="NC(C=O)CO",
        first_atom_idx=0,
        last_atom_idx=5,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    T=dict(
        resname="THR",
        smile="NC(C=O)C(O)C",
        first_atom_idx=0,
        last_atom_idx=6,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    W=dict(
        resname="TRP",
        smile="NC(C=O)CC1=CNC2=C1C=CC=C2",
        first_atom_idx=0,
        last_atom_idx=13,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    Y=dict(
        resname="TYR",
        smile="NC(C=O)CC1=CC=C(O)C=C1",
        first_atom_idx=0,
        last_atom_idx=11,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    V=dict(
        resname="VAL",
        smile="NC(C=O)C(C)C",
        first_atom_idx=0,
        last_atom_idx=6,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=(0, 1, 2),
    ),
    X=dict(
        resname="UNK",
        smile="NC(C=O)C",
        first_atom_idx=0,
        last_atom_idx=4,
        distogram_atom_idx=4,
        token_center_atom_idx=1,
        three_atom_indices_for_frame=None,
    ),
)

# nucleotides

DNA_NUCLEOTIDES = dict(
    A=dict(
        resname="DA",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O",
        first_atom_idx=0,
        last_atom_idx=21,
        complement="T",
        distogram_atom_idx=21,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    C=dict(
        resname="DC",
        smile="OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1O",
        first_atom_idx=0,
        last_atom_idx=19,
        complement="G",
        distogram_atom_idx=13,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    G=dict(
        resname="DG",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1O",
        first_atom_idx=0,
        last_atom_idx=22,
        complement="C",
        distogram_atom_idx=22,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    T=dict(
        resname="DT",
        smile="OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1O",
        first_atom_idx=0,
        last_atom_idx=20,
        complement="A",
        distogram_atom_idx=13,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=(11, 8, 6),
    ),
    X=dict(
        resname="DN",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O",
        first_atom_idx=0,
        last_atom_idx=21,
        complement="N",
        distogram_atom_idx=21,
        token_center_atom_idx=11,
        three_atom_indices_for_frame=None,
    ),
)

RNA_NUCLEOTIDES = dict(
    A=dict(
        resname="A",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=22,
        complement="U",
        distogram_atom_idx=22,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    C=dict(
        resname="C",
        smile="OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=20,
        complement="G",
        distogram_atom_idx=14,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    G=dict(
        resname="G",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=23,
        complement="C",
        distogram_atom_idx=23,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    U=dict(
        resname="U",
        smile="OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=20,
        complement="A",
        distogram_atom_idx=14,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=(12, 8, 6),
    ),
    X=dict(
        resname="N",
        smile="OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O",
        first_atom_idx=0,
        last_atom_idx=22,
        complement="N",
        distogram_atom_idx=22,
        token_center_atom_idx=12,
        three_atom_indices_for_frame=None,
    ),
)

# ligands

LIGANDS = dict(
    X=dict(
        resname="UNK",
        smile=".",
        first_atom_idx=0,
        last_atom_idx=0,
        distogram_atom_idx=0,
        token_center_atom_idx=0,
        three_atom_indices_for_frame=None,
    )
)

# complements in tensor form, following the ordering ACG(T|U)N

NUCLEIC_ACID_COMPLEMENT_TENSOR = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)

# some functions for nucleic acids


@typecheck
def reverse_complement(seq: str, nucleic_acid_type: Literal["dna", "rna"] = "dna"):
    """Get the reverse complement of a nucleic acid sequence."""
    if nucleic_acid_type == "dna":
        nucleic_acid_entries = DNA_NUCLEOTIDES
    elif nucleic_acid_type == "rna":
        nucleic_acid_entries = RNA_NUCLEOTIDES

    assert all(
        [nuc in nucleic_acid_entries for nuc in seq]
    ), "unknown nucleotide for given nucleic acid type"

    complement = [nucleic_acid_entries[nuc]["complement"] for nuc in seq]
    return "".join(complement[::-1])


@typecheck
def reverse_complement_tensor(t: Int[" n"]):  # type: ignore
    """Get the reverse complement of a nucleic acid sequence tensor."""
    complement = NUCLEIC_ACID_COMPLEMENT_TENSOR[t]
    reverse_complement = complement.flip(dims=(-1,))
    return reverse_complement


# metal ions

METALS = dict(
    Mg=dict(resname="Mg", smile="[Mg+2]"),
    Mn=dict(resname="Mn", smile="[Mn+2]"),
    Fe=dict(resname="Fe", smile="[Fe+3]"),
    Co=dict(resname="Co", smile="[Co+2]"),
    Ni=dict(resname="Ni", smile="[Ni+2]"),
    Cu=dict(resname="Cu", smile="[Cu+2]"),
    Zn=dict(resname="Zn", smile="[Zn+2]"),
    Na=dict(resname="Na", smile="[Na+]"),
    Cl=dict(resname="Cl", smile="[Cl-]"),
    Ca=dict(resname="Ca", smile="[Ca+2]"),
    K=dict(resname="K", smile="[K+]"),
)

# miscellaneous

MISC = dict(
    Phospholipid=dict(
        resname="UNL", smile="CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)O)OC(=O)CCCCCCCC1CC1CCCCCC"
    )
)

# atoms - for atom embeddings

ATOMS = ["C", "O", "N", "S", "P", *METALS]

assert is_unique(ATOMS)

# bonds for atom bond embeddings

ATOM_BONDS = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# PDB mmCIF to RDKit bond types

BOND_ORDER = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "AROM": Chem.BondType.AROMATIC,
}

assert is_unique(ATOM_BONDS)

# some rdkit helper function


@typecheck
def generate_conformation(mol: Mol) -> Mol:
    """Generate a conformation for a molecule."""
    mol = Chem.AddHs(mol)
    Chem.EmbedMultipleConfs(mol, numConfs=1)
    mol = Chem.RemoveHs(mol)
    return mol


@typecheck
def mol_from_smile(smile: str) -> Mol:
    """Generate an rdkit.Chem molecule from a SMILES string."""
    mol = Chem.MolFromSmiles(smile)
    return generate_conformation(mol)


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

        mol.AddBond(idx1, idx2, BOND_ORDER[order])

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


# initialize rdkit.Chem with canonical SMILES

CHAINABLE_BIOMOLECULES = [
    HUMAN_AMINO_ACIDS,
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
]

METALS_AND_MISC = [
    METALS,
    MISC,
]

for entries in [*CHAINABLE_BIOMOLECULES, *METALS_AND_MISC]:
    for rescode in entries:
        entry = entries[rescode]
        resname = entry["resname"]
        template_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chemical", f"{resname}.cif"
        )
        if os.path.exists(template_filepath):
            mol = mol_from_template_mmcif_file(template_filepath)
        else:
            mol = mol_from_smile(entry["smile"])
        entry["rdchem_mol"] = mol

for entries in CHAINABLE_BIOMOLECULES:
    for rescode in entries:
        entry = entries[rescode]
        mol = entry['rdchem_mol']
        num_atoms = mol.GetNumAtoms()

        assert 0 <= entry["first_atom_idx"] < num_atoms
        assert 0 <= entry["last_atom_idx"] < num_atoms
        assert 0 <= entry["distogram_atom_idx"] < num_atoms
        assert 0 <= entry["token_center_atom_idx"] < num_atoms

        if exists(entry.get('three_atom_indices_for_frame', None)):
            assert all([(0 <= i < num_atoms) for i in entry["three_atom_indices_for_frame"]])

        assert entry["first_atom_idx"] != entry["last_atom_idx"]
