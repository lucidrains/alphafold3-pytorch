import gzip
import os


def filter_pdb_lines(file_path: str, output_file_path: str):
    """Filter lines containing 'PDB' entries from a compressed `.dat.gz` file, and write them to an
    output file.

    :param file_path: Path to the compressed `.dat.gz` file to be read.
    :param output_file_path: Path to the output file where filtered lines will be written.
    """
    with gzip.open(file_path, "rt") as infile, open(output_file_path, "w") as outfile:
        # Run a generator expression to filter lines containing 'PDB'
        pdb_lines = (line for line in infile if "\tPDB\t" in line)
        outfile.writelines(pdb_lines)


if __name__ == "__main__":
    input_archive_file = "idmapping.dat.gz"
    output_file = os.path.join(
        "..", "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )
    filter_pdb_lines(input_archive_file, output_file)
