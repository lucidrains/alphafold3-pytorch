from typing import Iterator, Union
from torch.utils.data import Sampler
import polars as pl
import numpy as np


def get_chain_count(molecule_type) -> tuple[int, int, int]:
    """
    Returns the number of protein, nucleic acid, and ligand chains in a
    molecule based on its type.

    Example:
        n_prot, n_nuc, n_ligand = get_chain_count("protein")
    """
    match molecule_type:
        case "protein":
            return 1, 0, 0
        case "nucleic_acid":
            return 0, 1, 0
        case "ligand":
            return 0, 0, 1
        case "peptide":
            return 1, 0, 0
        case _:
            raise ValueError(f"Unknown molecule type: {molecule_type}")


def calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size) -> float:
    """
    Calculates the weight of a chain or an interface according to the formula
    provided in Section 2.5.1 of the AlphaFold3 supplementary information.
    """
    return (beta / cluster_size) * (
        alphas["prot"] * n_prot + alphas["nuc"] * n_nuc + alphas["ligand"] * n_ligand
    )


def get_chain_weight(molecule_type, cluster_size, alphas, beta) -> float:
    n_prot, n_nuc, n_ligand = get_chain_count(molecule_type)
    return calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size)


def get_interface_weight(
    molecule_type_1, molecule_type_2, cluster_size, alphas, beta
) -> float:
    p1, n1, l1 = get_chain_count(molecule_type_1)
    p2, n2, l2 = get_chain_count(molecule_type_2)

    n_prot = p1 + p2
    n_nuc = n1 + n2
    n_ligand = l1 + l2

    return calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size)


def get_cluster_sizes(mapping, cluster_id_col) -> dict[int, int]:
    """
    Returns a dictionary where keys are cluster IDs and values are the number
    of chains/interfaces in the cluster.
    """
    cluster_sizes = mapping.group_by(cluster_id_col).agg(pl.len()).sort(cluster_id_col)
    return {row[0]: row[1] for row in cluster_sizes.iter_rows()}


def compute_chain_weights(chains: pl.DataFrame, alphas, beta) -> pl.Series:
    molecule_idx = chains.get_column_index("molecule_id")
    cluster_idx = chains.get_column_index("cluster_id")
    cluster_sizes = get_cluster_sizes(chains, "cluster_id")

    return (
        chains.map_rows(
            lambda row: get_chain_weight(
                row[molecule_idx].split("-")[0],
                cluster_sizes[row[cluster_idx]],
                alphas,
                beta,
            ),
            return_dtype=pl.Float32,
        )
        .to_series(0)
        .rename("weight")
    )


def compute_interface_weights(interfaces: pl.DataFrame, alphas, beta) -> pl.Series:
    molecule_idx_1 = interfaces.get_column_index("interface_molecule_id_1")
    molecule_idx_2 = interfaces.get_column_index("interface_molecule_id_2")
    cluster_idx = interfaces.get_column_index("interface_cluster_id")
    cluster_sizes = get_cluster_sizes(interfaces, "interface_cluster_id")

    return (
        interfaces.map_rows(
            lambda row: get_interface_weight(
                row[molecule_idx_1].split("-")[0],
                row[molecule_idx_2].split("-")[0],
                cluster_sizes[row[cluster_idx]],
                alphas,
                beta,
            ),
            return_dtype=pl.Float32,
        )
        .to_series(0)
        .rename("weight")
    )


class WeightedSamplerPDB(Sampler[list[str]]):
    def __init__(
        self,
        chain_mapping_paths: Union[str, list[str]],
        interface_mapping_path: str,
        batch_size: int,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        alpha_prot: float = 3.0,
        alpha_nuc: float = 3.0,
        alpha_ligand: float = 1.0,
        pdb_ids_to_skip: list[str] = [],
    ):
        """
        Initializes a dataset for weighted sampling of PDB IDs.

        Args
        -------
            chain_mapping_paths (Union[str, list[str]])
                Path to the CSV file containing chain cluster
                mappings. If multiple paths are provided, they will be
                concatenated.
            interface_mapping_path (str)
                Path to the CSV file containing interface
                cluster mappings.
            batch_size (int)
                Number of PDB IDs to sample in each batch.
            beta_chain (float)
                Weighting factor for chain clusters.
            beta_interface (float)
                Weighting factor for interface clusters.
            alpha_prot (float)
                Weighting factor for protein chains.
            alpha_nuc (float)
                Weighting factor for nucleic acid chains.
            alpha_ligand (float)
                Weighting factor for ligand chains.
            pdb_ids_to_skip (list[str])
                List of PDB IDs to skip during sampling.
                Allow extra data filtering to ensure we avoid training
                on anomolous complexes that passed through all filtering
                and clustering steps.

        Example
        -------
        ```
        sampler = WeightedPDBSampler(...)
        for batch in sampler:
            print(batch)
        ```
        """

        # Load chain and interface mappings
        if not isinstance(chain_mapping_paths, list):
            chain_mapping_paths = [chain_mapping_paths]

        chain_mapping = [pl.read_csv(path) for path in chain_mapping_paths]
        chain_mapping = pl.concat(chain_mapping)
        interface_mapping = pl.read_csv(interface_mapping_path)

        # Filter out unwanted PDB IDs
        if len(pdb_ids_to_skip) > 0:
            chain_mapping = chain_mapping.filter(
                pl.col("pdb_id").is_in(pdb_ids_to_skip).not_()
            )
            interface_mapping = interface_mapping.filter(
                pl.col("pdb_id").is_in(pdb_ids_to_skip).not_()
            )

        # Calculate weights for chains and interfaces
        self.alphas = {"prot": alpha_prot, "nuc": alpha_nuc, "ligand": alpha_ligand}
        self.betas = {"chain": beta_chain, "interface": beta_interface}
        self.batch_size = batch_size

        chain_mapping.insert_column(
            len(chain_mapping.columns),
            compute_chain_weights(chain_mapping, self.alphas, self.betas["chain"]),
        )
        interface_mapping.insert_column(
            len(interface_mapping.columns),
            compute_interface_weights(
                interface_mapping, self.alphas, self.betas["interface"]
            ),
        )

        # Concatenate chain and interface mappings
        chain_mapping = chain_mapping.select(["pdb_id", "cluster_id", "weight"])

        num_chain_clusters = chain_mapping.get_column("cluster_id").max() + 1
        interface_mapping = interface_mapping.with_columns(
            (pl.col("interface_cluster_id") + num_chain_clusters).alias("cluster_id")
        )
        interface_mapping = interface_mapping.select(["pdb_id", "cluster_id", "weight"])
        self.mappings = chain_mapping.extend(interface_mapping)

        # Normalize weights
        self.weights = self.mappings.get_column("weight").to_numpy()
        self.weights = self.weights / self.weights.sum()

    def __len__(self) -> int:
        return len(self.mappings) // self.batch_size

    def __iter__(self) -> Iterator[list[str]]:
        while True:
            yield self.sample(self.batch_size)

    def sample(self, batch_size: int) -> list[str]:
        indices = np.random.choice(len(self.mappings), size=batch_size, p=self.weights)
        return self.mappings.get_column("pdb_id").gather(indices).to_list()

    def cluster_based_sample(self, batch_size: int) -> list[str]:
        """
        Samples PDB IDs based on cluster IDs. For each batch, a number of cluster IDs
        are selected randomly, and a PDB ID is sampled from each cluster based on the
        weights of the chains/interfaces in the cluster.

        Warning! Significantly slower than the regular `sample` method.
        """
        cluster_ids = self.mappings.get_column("cluster_id").unique().sample(batch_size)

        pdb_ids = []
        for cluster_id in cluster_ids:
            cluster = self.mappings.filter(pl.col("cluster_id") == cluster_id)
            if len(cluster) == 1:
                pdb_ids.append(cluster.item(0, "pdb_id"))
                continue
            cluster_weights = cluster.get_column("weight").to_numpy()
            cluster_weights = cluster_weights / cluster_weights.sum()
            idx = np.random.choice(len(cluster), p=cluster_weights)
            pdb_ids.append(cluster.item(idx, "pdb_id"))

        return pdb_ids


if __name__ == "__main__":
    interface_mapping_path = (
        "data/pdb_data/data_caches/clusterings/interface_cluster_mapping.csv"
    )
    chain_mapping_paths = [
        "data/pdb_data/data_caches/clusterings/ligand_chain_cluster_mapping.csv",
        "data/pdb_data/data_caches/clusterings/nucleic_acid_chain_cluster_mapping.csv",
        "data/pdb_data/data_caches/clusterings/peptide_chain_cluster_mapping.csv",
        "data/pdb_data/data_caches/clusterings/protein_chain_cluster_mapping.csv",
    ]

    dataset = WeightedSamplerPDB(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

    print(dataset.sample(64))
