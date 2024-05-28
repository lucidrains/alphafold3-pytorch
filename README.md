<img src="./alphafold3.png" width="500px"></img>

## Alphafold 3 - Pytorch (wip)

Implementation of <a href="https://www.nature.com/articles/s41586-024-07487-w">Alphafold 3</a> in Pytorch

Getting a fair number of emails. You can chat with me about this work <a href="https://discord.gg/x6FuzQPQXY">here</a>

## Appreciation

- <a href="https://github.com/joseph-c-kim">Joseph</a> for contributing the Relative Positional Encoding and the Smooth LDDT Loss!

- <a href="https://github.com/engelberger">Felipe</a> for contributing Weighted Rigid Align, Express Coordinates In Frame, Compute Alignment Error, and Centre Random Augmentation modules!

- <a href="https://github.com/amorehead">Alex</a> for fixing various issues in the transcribed algorithms

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
    dim_template_feats = 44
)

# mock inputs

seq_len = 16
atom_seq_len = seq_len * 27

atom_inputs = torch.randn(2, atom_seq_len, 77)
atom_lens = torch.randint(0, 27, (2, seq_len))
atompair_feats = torch.randn(2, atom_seq_len, atom_seq_len, 16)
additional_residue_feats = torch.randn(2, seq_len, 10)

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)
msa_mask = torch.ones((2, 7)).bool()

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)
residue_atom_indices = torch.randint(0, 27, (2, seq_len))

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
plddt_labels = torch.randint(0, 50, (2, seq_len))
resolved_labels = torch.randint(0, 2, (2, seq_len))

# train

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    residue_atom_lens = atom_lens,
    atompair_feats = atompair_feats,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    residue_atom_indices = residue_atom_indices,
    distance_labels = distance_labels,
    pae_labels = pae_labels,
    pde_labels = pde_labels,
    plddt_labels = plddt_labels,
    resolved_labels = resolved_labels
)

loss.backward()

# after much training ...

sampled_atom_pos = alphafold3(
    num_recycling_steps = 4,
    num_sample_steps = 16,
    atom_inputs = atom_inputs,
    residue_atom_lens = atom_lens,
    atompair_feats = atompair_feats,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

sampled_atom_pos.shape # (2, 16 * 27, 3)
```

## Contributing

At the project root, run

```bash
$ sh ./contribute.sh
```

Then, add your module to `alphafold3_pytorch/alphafold3.py`, add your tests to `tests/test_af3.py`, and submit a pull request. You can run the tests locally with

```bash
$ pytest tests/
```

## Docker

### Build Docker Container
```bash
docker build -t af3 .
```

### Run Container
```bash
## With GPUs
docker run  --gpus all -it af3
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
  month    = "May",
  year     =  2024
}
```

```bibtex
@inproceedings{Darcet2023VisionTN,
    title   = {Vision Transformers Need Registers},
    author  = {Timoth'ee Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:263134283}
}
```
