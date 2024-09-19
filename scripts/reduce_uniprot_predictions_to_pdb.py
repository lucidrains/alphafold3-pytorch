import glob
import gzip
import os
import shutil
from collections import defaultdict

import polars as pl
from tqdm import tqdm


def filter_pdb_files(
    input_archive_dir: str, output_dir: str, uniprot_to_pdb_id_mapping_filepath: str
):
    """Remove files from a given directory if they are not associated with a PDB entry, and extract
    to a given output directory all remaining archive files while grouping them their UniProt
    accession IDs.

    :param input_archive_dir: The path to the directory containing the input archive files.
    :param output_dir: The path to the directory where the filtered archive files will be saved.
    :param uniprot_to_pdb_id_mapping_filepath: The path to the file containing the mapping of
        Uniprot IDs to PDB IDs. This file is used to filter the archive files.
    """
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
        glob.glob(archive_file_pattern), desc="Filtering prediction files by PDB ID association"
    ):
        archive_accession_id = os.path.splitext(os.path.basename(archive_file))[0].split("-")[1]

        if archive_accession_id in uniprot_to_pdb_id_mapping:
            archives_to_keep[archive_accession_id].add(archive_file)

    for archive_accession_id in tqdm(
        archives_to_keep, desc="Extracting and grouping prediction files by accession ID"
    ):
        for archive in archives_to_keep[archive_accession_id]:
            output_subdir = os.path.join(output_dir, archive_accession_id)
            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(
                output_subdir, os.path.basename(archive).removesuffix(".gz")
            )
            with gzip.open(archive, "rb") as f_in, open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    input_archive_dir = os.path.join("data", "afdb_data", "unfiltered_train_mmcifs")
    output_dir = os.path.join("data", "afdb_data", "train_mmcifs")
    uniprot_to_pdb_id_mapping_filepath = os.path.join(
        "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )
    filter_pdb_files(input_archive_dir, output_dir, uniprot_to_pdb_id_mapping_filepath)
