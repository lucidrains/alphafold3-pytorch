<img src="./alphafold3.png" width="450px"></img>

## Alphafold 3 - Pytorch (wip)

Implementation of <a href="https://www.nature.com/articles/s41586-024-07487-w">Alphafold 3</a> in Pytorch

Getting a fair number of emails. You can chat with me about this work <a href="https://discord.gg/x6FuzQPQXY">here</a>

## Install

```bash
$ pip install alphafold3-pytorch
```

## Usage

```python
import torch
from alphafold3_pytorch import Alphafold3

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_additional_residue_feats = 33,
    dim_template_feats = 44
)

# mock inputs

seq_len = 16
atom_seq_len = seq_len * 27

atom_inputs = torch.randn(2, atom_seq_len, 77)
atom_mask = torch.ones((2, atom_seq_len)).bool()
atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
additional_residue_feats = torch.randn(2, seq_len, 33)

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)

# train

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atom_mask = atom_mask,
    atompair_feats = atompair_feats,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    templates = template_feats,
    template_mask = template_mask
)

loss.backward()

```

## Citations

```bibtex
@article{Abramson2024-fj,
  title    = "Accurate structure prediction of biomolecular interactions with
              {AlphaFold} 3",
  author   = "Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans,
              Richard and Green, Tim and Pritzel, Alexander and Ronneberger,
              Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick,
              Joshua and Bodenstein, Sebastian W and Evans, David A and Hung,
              Chia-Chun and O'Neill, Michael and Reiman, David and
              Tunyasuvunakool, Kathryn and Wu, Zachary and {\v Z}emgulyt{\.e},
              Akvil{\.e} and Arvaniti, Eirini and Beattie, Charles and
              Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and
              Congreve, Miles and Cowen-Rivers, Alexander I and Cowie, Andrew
              and Figurnov, Michael and Fuchs, Fabian B and Gladman, Hannah and
              Jain, Rishub and Khan, Yousuf A and Low, Caroline M R and Perlin,
              Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and
              Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine
              and Yakneen, Sergei and Zhong, Ellen D and Zielinski, Michal and
              {\v Z}{\'\i}dek, Augustin and Bapst, Victor and Kohli, Pushmeet
              and Jaderberg, Max and Hassabis, Demis and Jumper, John M",
  journal  = "Nature",
  month    =  may,
  year     =  2024
}
```
