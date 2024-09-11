import collections
import copy

import numpy as np
import polars as pl
import scipy
from beartype.typing import Dict, List

from alphafold3_pytorch.tensor_typing import typecheck
from alphafold3_pytorch.data.data_pipeline import GAP_ID
from alphafold3_pytorch.utils.utils import exists

# Constants

MSA_FEATURES = (
    "msa",
    "has_deletion",
    "deletion_value",
)

MSA_PAD_VALUES = {
    "msa_all_seq": GAP_ID,
    "msa_species_identifiers_all_seq": "",
    "has_deletion_all_seq": 0.0,
    "deletion_value_all_seq": 0.0,
    "profile_all_seq": 0.0,
    "deletion_mean_all_seq": 0.0,
    "msa": GAP_ID,
    "has_deletion": 0.0,
    "deletion_value": 0.0,
    "profile": 0.0,
    "deletion_mean": 0.0,
}


@typecheck
def _make_msa_df(chain_features: Dict[str, np.ndarray], gap_id: int = GAP_ID) -> pl.DataFrame:
    """
    Make DataFrame with MSA features needed for MSA pairing.
    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L118

    :param chain_features: The MSA chain feature dictionary.
    :return: The DataFrame with MSA features.
    """
    chain_msa = chain_features["msa_all_seq"]
    query_seq = chain_msa[0]
    per_seq_similarity = np.sum(query_seq[None] == chain_msa, axis=-1) / float(len(query_seq))
    per_seq_gap = np.sum(chain_msa == gap_id, axis=-1) / float(len(query_seq))
    msa_df = pl.DataFrame(
        {
            "msa_species_identifiers": chain_features["msa_species_identifiers_all_seq"],
            "msa_row": np.arange(len(chain_features["msa_species_identifiers_all_seq"])),
            "msa_similarity": per_seq_similarity,
            "gap": per_seq_gap,
        }
    )
    return msa_df


@typecheck
def _create_species_dict(msa_df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Create mapping from species to MSA dataframe of that species.
    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L137

    :param msa_df: The DataFrame with MSA features.
    :return: The mapping from species to MSA dataframe of that species.
    """
    species_lookup = {}
    for species_tuple, species_df in msa_df.group_by("msa_species_identifiers"):
        species_lookup[species_tuple[0]] = species_df
    return species_lookup


@typecheck
def _match_rows_by_sequence_similarity(
    this_species_msa_dfs: List[pl.DataFrame],
) -> List[np.ndarray]:
    """Find MSA sequence pairings across chains based on sequence similarity.

    Each chain's MSA sequences are first sorted by their sequence similarity to
    their respective target sequence. The sequences are then paired, starting
    from the sequences most similar to their target sequence.

    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L145

    :param this_species_msa_dfs: A list of DataFrames containing MSA features for sequences
        of a specific species.
    :return: A list of NumPy arrays, each containing M indices corresponding to paired MSA rows,
        where M is the number of chains.
    """
    all_paired_msa_rows = []

    num_seqs = [len(species_df) for species_df in this_species_msa_dfs if exists(species_df)]
    take_num_seqs = np.min(num_seqs)

    for species_df in this_species_msa_dfs:
        if exists(species_df):
            species_df_sorted = species_df.sort("msa_similarity", descending=True)
            msa_rows = species_df_sorted.get_column("msa_row").head(take_num_seqs).to_list()
        else:
            msa_rows = [-1] * take_num_seqs  # Take the last 'padding' row.
        all_paired_msa_rows.append(msa_rows)

    all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
    return all_paired_msa_rows


@typecheck
def pair_sequences(
    chains: List[Dict[str, np.ndarray]],
    max_num_species_sequences: int = 100,
) -> Dict[int, np.ndarray]:
    """
    Return indices for paired MSA sequences across chains.
    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L181

    :param chains: The MSA chain feature dictionaries.
    :param max_num_species_sequences: The maximum number of sequences to accept for any species.
    :return: The indices for paired MSA sequences across chains.
    """

    num_chains = len(chains)

    all_chain_species_dict = []
    common_species = set()
    for chain_features in chains:
        msa_df = _make_msa_df(chain_features)
        species_dict = _create_species_dict(msa_df)
        all_chain_species_dict.append(species_dict)
        common_species.update(set(species_dict))

    common_species = sorted(common_species)
    common_species.remove("-1")  # Remove target sequence species.

    all_paired_msa_rows = [np.zeros(len(chains), int)]
    all_paired_msa_rows_dict = {k: [] for k in range(num_chains)}
    all_paired_msa_rows_dict[num_chains] = [np.zeros(len(chains), int)]

    for species in common_species:
        if not species:
            continue
        this_species_msa_dfs = []
        species_dfs_present = 0
        for species_dict in all_chain_species_dict:
            if species in species_dict:
                this_species_msa_dfs.append(species_dict[species])
                species_dfs_present += 1
            else:
                this_species_msa_dfs.append(None)

        # Skip species that are present in only one chain.
        if species_dfs_present <= 1:
            continue

        # Skip species with too many sequences.
        if np.any(
            np.array(
                [
                    len(species_df)
                    for species_df in this_species_msa_dfs
                    if isinstance(species_df, pl.DataFrame)
                ]
            )
            > max_num_species_sequences
        ):
            continue

        paired_msa_rows = _match_rows_by_sequence_similarity(this_species_msa_dfs)
        all_paired_msa_rows.extend(paired_msa_rows)
        all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)

    all_paired_msa_rows_dict = {
        num_examples: np.array(paired_msa_rows)
        for num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
    }

    return all_paired_msa_rows_dict


@typecheck
def reorder_paired_rows(all_paired_msa_rows_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """Create a list of indices of paired MSA rows across chains.

    :param all_paired_msa_rows_dict: A mapping from the number of paired chains to the paired
        indices.
    :return: A NumPy array, with inner arrays containing indices of paired MSA rows across chains.
        The paired-index lists are ordered by: 1) the number of chains in the paired alignment,
        i.e, all-chain pairings will come first, and 2) e-values
    """
    all_paired_msa_rows = []

    for num_pairings in sorted(all_paired_msa_rows_dict, reverse=True):
        paired_rows = all_paired_msa_rows_dict[num_pairings]
        paired_rows_product = abs(np.array([np.prod(rows) for rows in paired_rows]))
        paired_rows_sort_index = np.argsort(paired_rows_product)
        all_paired_msa_rows.extend(paired_rows[paired_rows_sort_index])

    return np.array(all_paired_msa_rows)


@typecheck
def pad_features(feature: np.ndarray, feature_name: str) -> np.ndarray:
    """Add a 'padding' row at the end of the features list.

    The padding row will be selected as a 'paired' row in the case of partial
    alignment, for the chain that does not have paired alignment.

    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L91

    :param feature: The feature to be padded.
    :param feature_name: The name of the feature to be padded.
    :return: The feature with an additional padding row.
    """
    assert feature.dtype != np.dtype(np.string_)
    if feature_name in (
        "msa_all_seq",
        "has_deletion_all_seq",
        "deletion_value_all_seq",
    ):
        num_res = feature.shape[1]
        padding = MSA_PAD_VALUES[feature_name] * np.ones([1, num_res], feature.dtype)
    elif feature_name == "msa_species_identifiers_all_seq":
        padding = ["-1"]
    else:
        return feature
    feats_padded = np.concatenate([feature, padding], axis=0)
    return feats_padded


@typecheck
def copy_unpaired_features(chains: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """Copy unpaired features.

    :param chains: The MSA chain feature dictionaries.
    :return: The MSA chain feature dictionaries with unpaired features copied.
    """
    chain_keys = chains[0].keys()

    new_chains = []
    for chain in chains:
        new_chain_features = copy.deepcopy(chain)
        for feature_name in chain_keys:
            unpaired_feature_name = feature_name.removesuffix("_all_seq")
            if unpaired_feature_name in MSA_FEATURES:
                new_chain_features[unpaired_feature_name] = copy.deepcopy(chain[feature_name])

        new_chains.append(new_chain_features)

    return new_chains


@typecheck
def create_paired_features(
    chains: List[Dict[str, np.ndarray]],
) -> List[Dict[str, np.ndarray]]:
    """
    Pair MSA chain features.
    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L56

    :param chains: The MSA chain feature dictionaries.
    :return: The MSA chain feature dictionaries with paired features.
    """
    chain_keys = chains[0].keys()

    if len(chains) < 2:
        return chains
    else:
        updated_chains = []
        paired_chains_to_paired_row_indices = pair_sequences(chains)
        paired_rows = reorder_paired_rows(paired_chains_to_paired_row_indices)

        for chain_num, chain in enumerate(chains):
            new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}

            for feature_name in chain_keys:
                if feature_name in (
                    "profile_all_seq",
                    "deletion_mean_all_seq",
                ):
                    new_chain[feature_name] = chain[feature_name]
                elif feature_name.endswith("_all_seq"):
                    feats_padded = pad_features(chain[feature_name], feature_name)
                    new_chain[feature_name] = feats_padded[paired_rows[:, chain_num]]

            updated_chains.append(new_chain)

        return updated_chains


@typecheck
def merge_homomers_dense_msa(chains: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """Merge all identical chains, making the resulting MSA dense.

    :param chains: An iterable of features for each chain.
    :return: A list of feature dictionaries.  All features with the same entity_id
        will be merged - MSA features will be concatenated along the num_res
        dimension - making them dense.
    """
    entity_chains = collections.defaultdict(list)
    for chain in chains:
        entity_id = chain["entity_id"][0]
        entity_chains[entity_id].append(chain)

    grouped_chains = []
    for entity_id in sorted(entity_chains):
        chains = entity_chains[entity_id]
        grouped_chains.append(chains)

    chains = [
        merge_features_from_multiple_chains(chains, pair_msa_sequences=True)
        for chains in grouped_chains
    ]
    return chains


@typecheck
def block_diag(*arrs: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Construct a block diagonal like scipy.linalg.block_diag but with an optional padding value.

    :param arrs: The NumPy arrays to block diagonalize.
    :param pad_value: The padding value to use.
    :return: The block diagonalized NumPy array.
    """
    ones_arrs = [np.ones_like(x) for x in arrs]
    off_diag_mask = 1.0 - scipy.linalg.block_diag(*ones_arrs)
    diag = scipy.linalg.block_diag(*arrs)
    diag += (off_diag_mask * pad_value).astype(diag.dtype)
    return diag


@typecheck
def merge_features_from_multiple_chains(
    chains: List[Dict[str, np.ndarray]], pair_msa_sequences: bool
) -> Dict[str, np.ndarray]:
    """Merge features from multiple chains.

    :param chains: A list of feature dictionaries that we want to merge.
    :param pair_msa_sequences: Whether to concatenate MSA features along the num_res dimension (if
        True), or to block diagonalize them (if False).
    :return: A feature dictionary for the merged example.
    """
    chain_keys = chains[0].keys()

    chains_merged = {}
    for feature_name in chain_keys:
        feats = [x[feature_name] for x in chains]
        unpaired_feature_name = feature_name.removesuffix("_all_seq")

        if unpaired_feature_name in MSA_FEATURES:
            if pair_msa_sequences or "_all_seq" in feature_name:
                chains_merged[feature_name] = np.concatenate(feats, axis=-1)
            else:
                chains_merged[feature_name] = block_diag(
                    *feats, pad_value=MSA_PAD_VALUES[feature_name]
                )
        elif feature_name in ("profile_all_seq", "deletion_mean_all_seq"):
            chains_merged[feature_name] = np.concatenate(feats, axis=0)
        else:
            chains_merged[feature_name] = feats[0]

    return chains_merged


@typecheck
def concatenate_paired_and_unpaired_features(
    chains: Dict[str, np.ndarray],
    max_msas_per_chain: int | None = None,
) -> Dict[str, np.ndarray]:
    """Merge paired and unpaired features.

    :param chains: The MSA chain feature dictionaries.
    :param max_msas_per_chain: The maximum number of MSAs per chain.
    :return: The MSA chain feature dictionaries with paired and unpaired features merged.
    """
    if not exists(max_msas_per_chain):
        assert "msa" in chains and "msa_all_seq" in chains, (
            "If max_msas_per_chain is not provided, "
            "both 'msa' and 'msa_all_seq' must be present in 'chains'."
        )
        max_msas_per_chain = max(len(chains["msa"]), len(chains["msa_all_seq"])) * 2

    max_paired_msa_per_chain = max_msas_per_chain // 2

    for feature_name in MSA_FEATURES:
        if feature_name in chains:
            feat = chains[feature_name]
            feat_all_seq = chains[feature_name + "_all_seq"]
            chain_merged = np.concatenate([feat_all_seq[:max_paired_msa_per_chain], feat], axis=0)
            chains[feature_name + "_all_seq"] = chain_merged[:max_msas_per_chain]

            del chains[feature_name]

    return chains
