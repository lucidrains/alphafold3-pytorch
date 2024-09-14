import re
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

# functions

def join(arr, delimiter = ''): # just redo an ugly part of python
    return delimiter.join(arr)

# constants

aa_constants = get_residue_constants(res_chem_index = IS_PROTEIN)
restypes_index = dict(enumerate(aa_constants.restypes))

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

        sequence_data = [
            (
                f"molecule{i}",
                join([restypes_index.get(i, 'X') for i in ids]),
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

class ProstT5Wrapper(Module):
    def __init__(self):
        super().__init__()
        from transformers import T5Tokenizer, T5EncoderModel

        self.register_buffer('dummy', tensor(0), persistent = False)

        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case = False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.embed_dim = 1024

    def forward(
        self,
        aa_ids: Int['b n']
    ) -> Float['b n dpe']:

        device, seq_len = self.dummy.device, aa_ids.shape[-1]

        str_sequences = [
            join([restypes_index.get(i, 'X') for i in ids])
            for i, ids in enumerate(aa_ids)
        ]

        # following the readme at https://github.com/mheinzinger/ProstT5

        str_sequences = [join(list(re.sub(r"[UZOB]", "X", str_seq)), ' ') for str_seq in str_sequences]

        # encode to ids

        inputs = self.tokenizer.batch_encode_plus(
            str_sequences,
            add_special_tokens = True,
            padding = "longest",
            return_tensors = 'pt'
        ).to(device)

        # forward through plm

        embeddings = self.model(
            inputs.input_ids, 
            attention_mask = inputs.attention_mask
        )

        # remove prefix

        plm_embedding = embeddings.last_hidden_state[:, 1:(seq_len + 1)]
        return plm_embedding

# PLM embedding type and registry

PLMRegistry = dict(
    esm2_t33_650M_UR50D = partial(ESMWrapper, 'esm2_t33_650M_UR50D'),
    prostT5 = ProstT5Wrapper
)

PLMEmbedding = Literal[
    "esm2_t33_650M_UR50D",
    "prostT5"
]
