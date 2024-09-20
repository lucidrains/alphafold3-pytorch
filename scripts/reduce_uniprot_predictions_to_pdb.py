import glob
import gzip
import os
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import polars as pl
import timeout_decorator
from beartype.typing import Dict, Set, Tuple
from tqdm import tqdm

from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.utils.data_utils import extract_mmcif_metadata_field

PROCESS_ARCHIVE_MAX_SECONDS_PER_INPUT = 15


@timeout_decorator.timeout(PROCESS_ARCHIVE_MAX_SECONDS_PER_INPUT, use_signals=True)
def process_archive_with_timeout(archive_info: Tuple[str, Dict[str, Set[str]], str, str]):
    """Process a single archive file by extracting it to a given output directory and updating the
    release date of the associated PDB entries.

    :param archive_info: A tuple containing the path to the archive file, a dictionary mapping
        UniProt accession IDs to PDB IDs, the path to the input PDB directory, and the path to the
        output directory.
    """
    archive, uniprot_to_pdb_id_mapping, input_pdb_dir, output_dir = archive_info

    archive_accession_id = os.path.splitext(os.path.basename(archive))[0].split("-")[1]
    output_subdir = os.path.join(output_dir, archive_accession_id)

    pdb_release_date = datetime(1970, 1, 1)
    for pdb_id in list(uniprot_to_pdb_id_mapping[archive_accession_id]):
        pdb_id = pdb_id.lower()
        pdb_group_code = pdb_id[1:3]
        pdb_filepath = os.path.join(input_pdb_dir, pdb_group_code, f"{pdb_id}-assembly1.cif")

        if os.path.exists(pdb_filepath):
            try:
                mmcif_object = mmcif_parsing.parse_mmcif_object(
                    filepath=pdb_filepath, file_id=f"{pdb_id}-assembly1.cif"
                )
                mmcif_release_date = extract_mmcif_metadata_field(mmcif_object, "release_date")

                pdb_release_date = max(
                    pdb_release_date, datetime.strptime(mmcif_release_date, "%Y-%m-%d")
                )
            except Exception as e:
                print(
                    f"An error occurred while processing PDB ID {pdb_id} associated with {pdb_filepath}: {e}. Skipping this prediction..."
                )
                return

    if pdb_release_date == datetime(1970, 1, 1):
        print(
            f"Could not find PDB release date for {archive_accession_id}. Skipping this prediction..."
        )
        return

    os.makedirs(output_subdir, exist_ok=True)

    output_file = os.path.join(output_subdir, os.path.basename(archive).removesuffix(".gz"))
    with gzip.open(archive, "rb") as f_in, open(output_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    with open(output_file, "r") as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            if "_pdbx_audit_revision_history.revision_date" in line:
                new_lines.append(line)
                new_lines.append(f'"Structure model" 1 0 1 {pdb_release_date.date()} \n')
            else:
                new_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(new_lines)


def process_archive(archive_info: Tuple[str, Dict[str, Set[str]], str, str]):
    """Process a single archive file by extracting it to a given output directory and updating the
    release date of the associated PDB entries.

    :param archive_info: A tuple containing the path to the archive file, a dictionary mapping
        UniProt accession IDs to PDB IDs, the path to the input PDB directory, and the path to the
        output directory.
    """
    try:
        process_archive_with_timeout(archive_info)
    except Exception as e:
        print(
            f"Processing of archive info {archive_info} took too long and was terminated due to: {e}. Skipping this prediction..."
        )


def filter_pdb_files(
    input_archive_dir: str,
    input_pdb_dir: str,
    output_dir: str,
    uniprot_to_pdb_id_mapping_filepath: str,
):
    """Remove files from a given directory if they are not associated with a PDB entry, and extract
    to a given output directory all remaining archive files while grouping them their UniProt
    accession IDs."""
    os.makedirs(output_dir, exist_ok=True)

    uniprot_to_pdb_id_mapping_df = pl.read_csv(
        uniprot_to_pdb_id_mapping_filepath,
        has_header=False,
        separator="\t",
        new_columns=["uniprot_accession", "database", "pdb_id"],
    )
    uniprot_to_pdb_id_mapping_df.drop_in_place("database")

    uniprot_to_pdb_id_mapping = defaultdict(set)
    for row in uniprot_to_pdb_id_mapping_df.iter_rows():
        uniprot_to_pdb_id_mapping[row[0]].add(row[1])

    archives_to_keep = defaultdict(set)
    archive_file_pattern = os.path.join(input_archive_dir, "*model_v4.cif.gz")

    for archive_file in tqdm(
        glob.glob(archive_file_pattern),
        desc="Filtering prediction files by PDB ID association",
    ):
        archive_accession_id = os.path.splitext(os.path.basename(archive_file))[0].split("-")[1]

        if archive_accession_id in uniprot_to_pdb_id_mapping:
            archives_to_keep[archive_accession_id].add(archive_file)

    # Prepare the multiprocessing pool
    pool = Pool(processes=12)

    # Prepare arguments for each worker
    archive_infos = [
        (archive, uniprot_to_pdb_id_mapping, input_pdb_dir, output_dir)
        for accession_id in archives_to_keep
        for archive in archives_to_keep[accession_id]
    ]

    # Process archives in parallel
    for _ in tqdm(
        pool.imap_unordered(process_archive, archive_infos),
        total=len(archive_infos),
        desc="Processing archives",
    ):
        pass

    pool.close()
    pool.join()


if __name__ == "__main__":
    input_archive_dir = os.path.join("data", "afdb_data", "unfiltered_train_mmcifs")
    input_pdb_dir = os.path.join("data", "pdb_data", "train_mmcifs")
    output_dir = os.path.join("data", "afdb_data", "train_mmcifs")
    uniprot_to_pdb_id_mapping_filepath = os.path.join(
        "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )
    filter_pdb_files(
        input_archive_dir, input_pdb_dir, output_dir, uniprot_to_pdb_id_mapping_filepath
    )
