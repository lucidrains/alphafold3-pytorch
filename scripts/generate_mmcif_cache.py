import argparse
import json
import os
from argparse import Namespace
from functools import partial
from loguru import logger
from multiprocessing import Pool
from typing import Dict, Optional

from tqdm import tqdm

from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.tensor_typing import typecheck


@typecheck
def parse_file(
    f: str, args: Namespace, chain_cluster_size_dict: Optional[Dict[str, int]] = None
) -> Dict[str, Dict]:
    """
    Parse a single mmCIF file and return a dictionary with the parsed data.

    :param f: The filename of the mmCIF file
    :param args: The command line arguments
    :param chain_cluster_size_dict: A dictionary mapping chain IDs to cluster sizes
    :return: A dictionary with the parsed data
    """
    with open(os.path.join(args.mmcif_dir, f), "r") as fp:
        mmcif_string = fp.read()
    file_id = os.path.splitext(f)[0]
    parsing_result = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string)
    if parsing_result.mmcif_object is None:
        logger.info(f"Could not parse {f}. Skipping...")
        return {}
    else:
        parsing_result = parsing_result.mmcif_object

    local_data = {}
    local_data["release_date"] = parsing_result.header["release_date"]

    chain_ids, seqs = list(zip(*parsing_result.chain_to_seqres.items()))

    if chain_cluster_size_dict is not None:
        cluster_sizes = []
        for chain_id in chain_ids:
            full_name = "_".join([file_id, chain_id])
            cluster_size = chain_cluster_size_dict.get(full_name.upper(), -1)
            cluster_sizes.append(cluster_size)

        local_data["cluster_sizes"] = cluster_sizes

    local_data["chain_ids"] = chain_ids
    local_data["seqs"] = seqs
    local_data["no_chains"] = len(chain_ids)

    local_data["resolution"] = parsing_result.header["resolution"]

    return {file_id: local_data}


@typecheck
def main(args: Namespace):
    chain_cluster_size_dict = None
    if args.cluster_file is not None:
        chain_cluster_size_dict = {}
        with open(args.cluster_file, "r") as fp:
            clusters = [line.strip() for line in fp.readlines()]

        for cluster in clusters:
            chain_ids = cluster.split()
            cluster_len = len(chain_ids)
            for chain_id in chain_ids:
                chain_id = chain_id.upper()
                chain_cluster_size_dict[chain_id] = cluster_len

    files = [f for f in os.listdir(args.mmcif_dir) if ".cif" in f]
    fn = partial(parse_file, args=args, chain_cluster_size_dict=chain_cluster_size_dict)
    data = {}
    with Pool(processes=args.no_workers) as p:
        with tqdm(total=len(files)) as pbar:
            for d in p.imap_unordered(fn, files, chunksize=args.chunksize):
                data.update(d)
                pbar.update()

    with open(args.output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mmcif_dir", type=str, help="Directory containing mmCIF files")
    parser.add_argument("output_path", type=str, help="Path for .json output")
    parser.add_argument(
        "--no_workers", type=int, default=4, help="Number of workers to use for parsing"
    )
    parser.add_argument(
        "--cluster_file",
        type=str,
        default=None,
        help=(
            "Path to a cluster file (e.g. PDB40), one cluster "
            "({COMP1_ID}_{CHAIN_ID} {COMP1_ID}_{CHAIN_ID} ...) per line. "
            "Chains not in this cluster file will NOT be filtered by cluster "
            "size."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10,
        help="How many files should be distributed to each worker at a time",
    )

    args = parser.parse_args()

    main(args)
