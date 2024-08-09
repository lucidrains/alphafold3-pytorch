# %% [markdown]
# # Curating AlphaFold 3 PDB Evaluation Dataset
#
# For evaluating trained AlphaFold 3 models, we propose a modified (i.e., more stringent) version of the
# evaluation procedure outlined in Abramson et al (2024).
#
# The recent PDB evaluation set construction started by taking all PDB entries released between 2023-01-14 and
# 2024-04-30, a date range falling after any data in our training or validation sets which had maximum release dates
# of 2021-09-30 and 2023-01-13, respectively.
# Each entry in the date range was expanded from the asymmetric unit to Biological Assembly 1, then two filters were
# applied:
# • Filtering to non-NMR entries with resolution better than 4.5 Å.
# • Filtering to complexes with less than 5,120 tokens under our tokenization scheme (see subsection 2.6 of the AF3 supplement).
# Predictions on the recent PDB set were made on the full post-assembly complex, but crystallization aids (Table 9)
# were removed from the complex for prediction and scoring, along with all bonds for structures with homomeric sub-
# complexes lacking the corresponding homomeric symmetry.
# ... (see the PDB evaluation set clustering script)
#

# %%
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import timeout_decorator
from tqdm.contrib.concurrent import process_map

from alphafold3_pytorch.common.paper_constants import (
    CRYSTALLOGRAPHY_METHODS,
    LIGAND_EXCLUSION_SET,
)
from alphafold3_pytorch.data import mmcif_parsing, mmcif_writing
from alphafold3_pytorch.data.mmcif_parsing import MmcifObject
from alphafold3_pytorch.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists
from scripts.filter_pdb_train_mmcifs import (
    FILTER_STRUCTURE_MAX_SECONDS_PER_INPUT,
    filter_pdb_release_date,
    filter_resolution,
    impute_missing_assembly_metadata,
    remove_crystallization_aids,
    remove_excluded_ligands,
    remove_hydrogens,
)
from scripts.filter_pdb_val_mmcifs import filter_num_tokens

# Helper functions


@typecheck
def filter_experiment_type(mmcif_object: MmcifObject, types_to_ignore: List[str]) -> bool:
    """Filter based on experiment type."""
    return (
        "structure_method" in mmcif_object.header
        and exists(mmcif_object.header["structure_method"])
        and not any(
            type_to_ignore in mmcif_object.header["structure_method"].upper()
            for type_to_ignore in types_to_ignore
        )
    )


@typecheck
def prefilter_target(
    mmcif_object: MmcifObject,
    min_cutoff_date: datetime = datetime(2023, 1, 14),
    max_cutoff_date: datetime = datetime(2024, 4, 30),
    experiment_types_to_ignore: List[str] = ["NMR"],
    max_resolution: float = 4.5,
    max_tokens: int = 5120,
) -> MmcifObject | None:
    """Pre-filter a target based on various criteria."""
    target_passes_prefilters = (
        filter_pdb_release_date(
            mmcif_object, min_cutoff_date=min_cutoff_date, max_cutoff_date=max_cutoff_date
        )
        and filter_experiment_type(mmcif_object, types_to_ignore=experiment_types_to_ignore)
        and filter_resolution(mmcif_object, max_resolution=max_resolution, exclusive_max=True)
        and filter_num_tokens(mmcif_object, max_tokens=max_tokens, exclusive_max=True)
    )
    return mmcif_object if target_passes_prefilters else None


@typecheck
@timeout_decorator.timeout(FILTER_STRUCTURE_MAX_SECONDS_PER_INPUT, use_signals=False)
def filter_structure_with_timeout(
    filepath: str,
    output_dir: str,
    min_cutoff_date: datetime = datetime(2023, 1, 14),
    max_cutoff_date: datetime = datetime(2024, 4, 30),
    keep_ligands_in_exclusion_set: bool = False,
):
    """
    Given an input mmCIF file, create a new filtered mmCIF file
    using AlphaFold 3's PDB evaluation dataset filtering criteria under a
    timeout constraint.
    """
    # Section 2.5.4 of the AlphaFold 3 supplement
    asym_filepath = os.path.join(
        os.path.dirname(filepath).replace("unfiltered_assembly", "unfiltered_asym"),
        os.path.basename(filepath).replace("-assembly1", ""),
    )
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    output_file_dir = os.path.join(output_dir, file_id[1:3])
    output_filepath = os.path.join(output_file_dir, f"{file_id}.cif")
    os.makedirs(output_file_dir, exist_ok=True)

    # Filtering of targets
    mmcif_object = mmcif_parsing.parse_mmcif_object(filepath, file_id)
    asym_mmcif_object = mmcif_parsing.parse_mmcif_object(asym_filepath, file_id)
    # NOTE: The assembly mmCIF does not contain the full header or the
    # structure connectivity (i.e., bond) information, so we impute
    # these from the asymmetric unit mmCIF
    mmcif_object = impute_missing_assembly_metadata(mmcif_object, asym_mmcif_object)
    mmcif_object = prefilter_target(
        mmcif_object,
        min_cutoff_date=min_cutoff_date,
        max_cutoff_date=max_cutoff_date,
        experiment_types_to_ignore=["NMR"],
        max_resolution=4.5,
        max_tokens=5120,
    )
    if not exists(mmcif_object):
        print(f"Skipping target due to prefiltering: {file_id}")
        return
    # Filtering of bioassemblies
    # NOTE: Here, we remove waters even though the AlphaFold 3 supplement doesn't mention removing them during filtering
    mmcif_object = remove_hydrogens(mmcif_object, remove_waters=True)
    if not keep_ligands_in_exclusion_set:
        # NOTE: The AlphaFold 3 supplement suggests the validation and evaluation datasets remove these (excluded) ligands
        mmcif_object = remove_excluded_ligands(mmcif_object, LIGAND_EXCLUSION_SET)
    mmcif_object = remove_crystallization_aids(mmcif_object, CRYSTALLOGRAPHY_METHODS)
    if len(mmcif_object.chains_to_remove) < len(mmcif_object.structure):
        # Save a filtered structure as an mmCIF file along with its latest metadata
        mmcif_object = mmcif_parsing.filter_mmcif(mmcif_object)
        mmcif_writing.write_mmcif(
            mmcif_object,
            output_filepath,
            gapless_poly_seq=True,
            insert_orig_atom_names=True,
            insert_alphafold_mmcif_metadata=False,
        )
        print(f"Finished filtering structure: {mmcif_object.file_id}")


@typecheck
def filter_structure(args: Tuple[str, str, datetime, datetime, bool]):
    """
    Given an input mmCIF file, create a new filtered mmCIF file
    using AlphaFold 3's PDB evaluation dataset filtering criteria.
    """
    filepath, output_dir, min_cutoff_date, max_cutoff_date, keep_ligands_in_exclusion_set = args
    file_id = os.path.splitext(os.path.basename(filepath))[0]
    output_file_dir = os.path.join(output_dir, file_id[1:3])
    output_filepath = os.path.join(output_file_dir, f"{file_id}.cif")

    try:
        filter_structure_with_timeout(
            filepath,
            output_dir,
            min_cutoff_date=min_cutoff_date,
            max_cutoff_date=max_cutoff_date,
            keep_ligands_in_exclusion_set=keep_ligands_in_exclusion_set,
        )
    except Exception as e:
        print(f"Skipping structure filtering of {filepath} due to: {e}")
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
            except Exception as e:
                print(
                    f"Failed to remove partially filtered file {output_filepath} due to: {e}. Skipping its removal..."
                )


if __name__ == "__main__":
    # Parse command-line arguments

    parser = argparse.ArgumentParser(
        description="Filter mmCIF files to curate the AlphaFold 3 PDB evaluation dataset."
    )
    parser.add_argument(
        "-i",
        "--mmcif_assembly_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "unfiltered_assembly_mmcifs"),
        help="Path to the input directory containing `assembly1` mmCIF files to filter.",
    )
    parser.add_argument(
        "-a",
        "--mmcif_asym_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "unfiltered_asym_mmcifs"),
        help="Path to the input directory containing asymmetric unit mmCIF files with which to filter the `assembly1` mmCIF files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "test_mmcifs"),
        help="Path to the output directory in which to store filtered mmCIF dataset files.",
    )
    parser.add_argument(
        "-f",
        "--min_cutoff_date",
        type=lambda t: datetime.strptime(t, "%Y-%m-%d"),
        default=datetime(2023, 1, 14),
        help="Minimum cutoff date for filtering PDB release dates.",
    )
    parser.add_argument(
        "-l",
        "--max_cutoff_date",
        type=lambda t: datetime.strptime(t, "%Y-%m-%d"),
        default=datetime(2024, 4, 30),
        help="Maximum cutoff date for filtering PDB release dates.",
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip filtering of existing output files.",
    )
    parser.add_argument(
        "-e",
        "--keep_ligands_in_exclusion_set",
        action="store_true",
        help="Keep ligands in the exclusion set during filtering.",
    )
    parser.add_argument(
        "-n",
        "--no_workers",
        type=int,
        default=16,
        help="Number of workers to use for filtering.",
    )
    parser.add_argument(
        "-w",
        "--chunksize",
        type=int,
        default=1,
        help="How many files should be distributed to each worker at a time.",
    )
    args = parser.parse_args()

    assert os.path.exists(
        args.mmcif_assembly_dir
    ), f"Input assembly directory {args.mmcif_assembly_dir} does not exist."
    assert os.path.exists(
        args.mmcif_asym_dir
    ), f"Input asymmetric unit directory {args.mmcif_asym_dir} does not exist."

    # Filter structures across all worker processes

    args_tuples = [
        (
            filepath,
            args.output_dir,
            args.min_cutoff_date,
            args.max_cutoff_date,
            args.keep_ligands_in_exclusion_set,
        )
        for filepath in glob.glob(os.path.join(args.mmcif_assembly_dir, "*", "*.cif"))
        if "assembly1" in os.path.basename(filepath)
        and os.path.exists(
            os.path.join(
                os.path.dirname(filepath).replace("unfiltered_assembly", "unfiltered_asym"),
                os.path.basename(filepath).replace("-assembly1", ""),
            )
        )
        and not (
            args.skip_existing
            and os.path.exists(
                os.path.join(
                    args.output_dir,
                    os.path.splitext(os.path.basename(filepath))[0][1:3],
                    f"{os.path.splitext(os.path.basename(filepath))[0]}.cif",
                )
            )
        )
    ]
    process_map(
        filter_structure,
        args_tuples,
        max_workers=args.no_workers,
        chunksize=args.chunksize,
    )
