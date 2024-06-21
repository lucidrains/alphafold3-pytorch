<img src="./alphafold3.png" width="500px"></img>

## Alphafold 3 - Pytorch

Implementation of <a href="https://www.nature.com/articles/s41586-024-07487-w">Alphafold 3</a> in Pytorch

You can chat with other researchers about this work <a href="https://discord.gg/x6FuzQPQXY">here</a>

<a href="https://www.youtube.com/watch?v=qjFgthkKxcA">Review of the paper</a> by <a href="https://x.com/sokrypton">Sergey</a>

A fork with full Lightning + Hydra support is being maintained by <a href="https://github.com/amorehead">Alex</a> at <a href="https://github.com/amorehead/alphafold3-pytorch-lightning-hydra">this repository</a>

## Appreciation

- <a href="https://github.com/joseph-c-kim">Joseph</a> for contributing the Relative Positional Encoding and the Smooth LDDT Loss!

- <a href="https://github.com/engelberger">Felipe</a> for contributing Weighted Rigid Align, Express Coordinates In Frame, Compute Alignment Error, and Centre Random Augmentation modules!

- <a href="https://github.com/amorehead">Alex</a> for fixing various issues in the transcribed algorithms

- <a href="https://github.com/gitabtion">Heng</a> for pointing out inconsistencies with the paper and pull requesting the solutions

- <a href="https://github.com/luwei0917">Wei Lu</a> for catching a few erroneous hyperparameters

- <a href="https://github.com/amorehead">Alex</a> for the PDB dataset preparation script!

- <a href="https://github.com/patrick-kidger">Patrick</a> for <a href="https://docs.kidger.site/jaxtyping/">jaxtyping</a>, <a href="https://github.com/fferflo">Florian</a> for <a href="https://github.com/fferflo/einx">einx</a>, and of course, <a href="https://github.com/arogozhnikov">Alex</a> for <a href="https://einops.rocks/">einops</a>

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
molecule_atom_lens = torch.randint(1, 3, (2, seq_len))
atom_seq_len = molecule_atom_lens.sum(dim = -1).amax()

atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)

additional_molecule_feats = torch.randn(2, seq_len, 9)
molecule_ids = torch.randint(0, 32, (2, seq_len))

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)
msa_mask = torch.ones((2, 7)).bool()

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)
molecule_atom_indices = molecule_atom_lens - 1 # last atom, as an example

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
plddt_labels = torch.randint(0, 50, (2, seq_len))
resolved_labels = torch.randint(0, 2, (2, seq_len))

# train

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    molecule_atom_indices = molecule_atom_indices,
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
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

sampled_atom_pos.shape # (2, <atom_seqlen>, 3)
```

## Data preparation

To acquire the AlphaFold 3 PDB dataset, first download all complexes in the Protein Data Bank (PDB), and then preprocess them with the script referenced below. The PDB can be downloaded from the RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb. The script below assumes you have downloaded the PDB in the **mmCIF file format** (e.g., placing it at `data/mmCIF/` by default). On the RCSB website, navigate down to "Download Protocols", and follow the download instructions depending on your location.

> WARNING: Downloading PDB can take up to 1TB of space.

After downloading, you should have a directory formatted like this:
https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/
```bash
00/
01/
02/
..
zz/
```

In this directory, unzip all the files:
```bash
find . -type f -name "*.gz" -exec gzip -d {} \;
```

Next run the commands `wget -P data/CCD/ https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz` and `wget -P data/CCD/ https://files.wwpdb.org/pub/pdb/data/component-models/complete/chem_comp_model.cif.gz` from the project's root directory to download the latest version of the PDB's Chemical Component Dictionary (CCD) and its structural models. Extract each of these files using the command `find data/CCD/ -type f -name "*.gz" -exec gzip -d {} \;`.

Then run the following with <pdb_dir>, <ccd_dir>, and <output_dir> replaced with the locations of your local copies of the PDB, CCD, and your desired dataset output directory (e.g., `data/PDB_set/` by default).
```bash
python alphafold3_pytorch/pdb_dataset_curation.py --mmcif_dir <pdb_dir> --ccd_dir <ccd_dir> --output_dir <output_dir>
```

See the script for more options. Each mmCIF that successfully passes
all processing steps will be written to <output_dir> within a subdirectory
named using the mmCIF's second and third PDB ID characters (e.g. `5c`).

## Contributing

At the project root, run

```bash
$ sh ./contribute.sh
```

Then, add your module to `alphafold3_pytorch/alphafold3.py`, add your tests to `tests/test_af3.py`, and submit a pull request. You can run the tests locally with

```bash
$ pytest tests/
```

## Docker Image
The included `Dockerfile` contains the required dependencies to run the package and to train/inference using PyTorch with GPUs.

The default base image is `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` and installs the latest version of this package from the `main` GitHub branch.

```bash
## Build Docker Container
docker build -t af3 .
```

Alternatively, use build arguments to rebuild the image with different software versions:
- `PYTORCH_TAG`: Changes the base image and thus builds with different PyTorch, CUDA, and/or cuDNN versions.
- `GIT_TAG`: Changes the tag of this repo to clone and install the package.

For example:
```bash
## Use build argument to change versions
docker build --build-arg "PYTORCH_TAG=2.2.1-cuda12.1-cudnn8-devel" --build-arg "GIT_TAG=0.1.15" -t af3 .
```

Then, run the container with GPUs and mount a local volume (for training) using the following command:

```bash
## Run Container
docker run -v .:/data --gpus all -it af3
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

```bibtex
@article{Arora2024SimpleLA,
    title   = {Simple linear attention language models balance the recall-throughput tradeoff},
    author  = {Simran Arora and Sabri Eyuboglu and Michael Zhang and Aman Timalsina and Silas Alberti and Dylan Zinsley and James Zou and Atri Rudra and Christopher R'e},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.18668},
    url     = {https://api.semanticscholar.org/CorpusID:268063190}
}
```

```bibtex
@article{Puny2021FrameAF,
    title   = {Frame Averaging for Invariant and Equivariant Network Design},
    author  = {Omri Puny and Matan Atzmon and Heli Ben-Hamu and Edward James Smith and Ishan Misra and Aditya Grover and Yaron Lipman},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2110.03336},
    url     = {https://api.semanticscholar.org/CorpusID:238419638}
}
```

```bibtex
@article{Duval2023FAENetFA,
    title   = {FAENet: Frame Averaging Equivariant GNN for Materials Modeling},
    author  = {Alexandre Duval and Victor Schmidt and Alex Hernandez Garcia and Santiago Miret and Fragkiskos D. Malliaros and Yoshua Bengio and David Rolnick},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2305.05577},
    url     = {https://api.semanticscholar.org/CorpusID:258564608}
}
```

```bibtex
@article{Wang2022DeepNetST,
    title   = {DeepNet: Scaling Transformers to 1, 000 Layers},
    author  = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Furu Wei},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2203.00555},
    url     = {https://api.semanticscholar.org/CorpusID:247187905}
}
```

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```
