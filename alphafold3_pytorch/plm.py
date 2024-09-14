from functools import partial

import torch
from torch import tensor
from torch.nn import Module

from beartype.typing import Literal

from alphafold3_pytorch.tensor_typing import (
    typecheck,
    Float,
    Int
)

from alphafold3_pytorch.common.biomolecule import (
    get_residue_constants,
)

from alphafold3_pytorch.inputs import (
    IS_PROTEIN,
)

# class

class ESMWrapper(Module):
    def __init__(
        self,
        esm_name,
        repr_layer = 33
    ):
        super().__init__()
        import esm
        self.repr_layer = repr_layer
        self.model, alphabet = esm.pretrained.load_model_and_alphabet_hub(esm_name)
        self.batch_converter = alphabet.get_batch_converter()

        self.embed_dim = self.model.embed_dim
        self.register_buffer('dummy', tensor(0), persistent = False)

    @torch.no_grad()
    @typecheck
    def forward(
        self,
        aa_ids: Int['b n']
    ) -> Float['b n dpe']:

        device, repr_layer = self.dummy.device, self.repr_layer

        aa_constants = get_residue_constants(res_chem_index=IS_PROTEIN)
        sequence_data = [
            (
                f"molecule{i}",
                "".join(
                    [
                        (
                            aa_constants.restypes[id]
                            if 0 <= id < len(aa_constants.restypes)
                            else "X"
                        )
                        for id in ids
                    ]
                ),
            )
            for i, ids in enumerate(aa_ids)
        ]

        _, _, batch_tokens = self.batch_converter(sequence_data)
        batch_tokens = batch_tokens.to(device)

        self.model.eval()
        results = self.model(batch_tokens, repr_layers=[repr_layer])

        token_representations = results["representations"][repr_layer]

        sequence_representations = []
        for i, (_, seq) in enumerate(sequence_data):
            sequence_representations.append(token_representations[i, 1 : len(seq) + 1])
        plm_embeddings = torch.stack(sequence_representations, dim=0)

        return plm_embeddings

# PLM embedding type and registry

PLMEmbedding = Literal["esm2_t33_650M_UR50D"]

PLMRegistry = dict(
    esm2_t33_650M_UR50D = partial(ESMWrapper, 'esm2_t33_650M_UR50D')
)
