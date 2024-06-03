from __future__ import annotations

from functools import wraps, partial
from pathlib import Path

from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.attention import pad_at_dim

from typing import TypedDict, List, Callable

from alphafold3_pytorch.typing import (
    typecheck,
    beartype_isinstance,
    Int, Bool, Float
)

import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader as OrigDataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from ema_pytorch import EMA

from lightning import Fabric

# constants

@typecheck
class AtomInput(TypedDict):
    atom_inputs:                Float['m dai']
    molecule_atom_lens:         Int[' n']
    atompair_inputs:            Float['m m dapi'] | Float['nw w (w*2) dapi']
    additional_molecule_feats:  Float['n 10']
    templates:                  Float['t n n dt']
    msa:                        Float['s n dm']
    template_mask:              Bool[' t'] | None
    msa_mask:                   Bool[' s'] | None
    atom_pos:                   Float['m 3'] | None
    molecule_atom_indices:      Int[' n'] | None
    distance_labels:            Int['n n'] | None
    pae_labels:                 Int['n n'] | None
    pde_labels:                 Int[' n'] | None
    resolved_labels:            Int[' n'] | None

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

@typecheck
def accum_dict(
    past_losses: dict | None,
    losses: dict,
    scale: float = 1.
):
    losses = {k: v * scale for k, v in losses.items()}

    if not exists(past_losses):
        return losses

    for loss_name in past_losses.keys():
        past_losses[loss_name] += losses.get(loss_name, 0.)

    return past_losses

# dataloader and collation fn

@typecheck
def collate_af3_inputs(
    inputs: List,
    int_pad_value = -1,
    map_input_fn: Callable | None = None
):

    if exists(map_input_fn):
        inputs = [map_input_fn(i) for i in inputs]

    # make sure all inputs are AtomInput

    assert all([beartype_isinstance(i, AtomInput) for i in inputs])

    # separate input dictionary into keys and values

    keys = inputs[0].keys()
    inputs = [i.values() for i in inputs]

    outputs = []

    for grouped in zip(*inputs):
        # if all None, just return None

        not_none_grouped = [*filter(exists, grouped)]

        if len(not_none_grouped) == 0:
            outputs.append(None)
            continue

        # default to empty tensor for any Nones

        one_tensor = not_none_grouped[0]

        dtype = one_tensor.dtype
        ndim = one_tensor.ndim

        # use -1 for padding int values, for assuming int are labels - if not, handle within alphafold3

        if dtype in (torch.int, torch.long):
            pad_value = int_pad_value
        elif dtype == torch.bool:
            pad_value = False
        else:
            pad_value = 0.

        # get the max lengths across all dimensions

        shapes_as_tensor = torch.stack([Tensor(tuple(g.shape) if exists(g) else ((0,) * ndim)).int() for g in grouped], dim = -1)

        max_lengths = shapes_as_tensor.amax(dim = -1)

        default_tensor = torch.full(max_lengths.tolist(), pad_value, dtype = dtype)

        # pad across all dimensions

        padded_inputs = []

        for inp in grouped:

            if not exists(inp):
                padded_inputs.append(default_tensor)
                continue

            for dim, max_length in enumerate(max_lengths.tolist()):
                inp = pad_at_dim(inp, (0, max_length - inp.shape[dim]), value = pad_value, dim = dim)

            padded_inputs.append(inp)

        # stack

        stacked = torch.stack(padded_inputs)

        outputs.append(stacked)

    # reconstitute dictionary

    return dict(tuple(zip(keys, outputs)))

@typecheck
def DataLoader(
    *args,
    map_input_fn: Callable | None = None,
    **kwargs
):
    collate_fn = collate_af3_inputs

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn = map_input_fn)

    return OrigDataLoader(*args, collate_fn = collate_fn, **kwargs)

# default scheduler used in paper w/ warmup

def default_lambda_lr_fn(steps):
    # 1000 step warmup

    if steps < 1000:
        return steps / 1000

    # decay 0.95 every 5e4 steps

    steps -= 1000
    return 0.95 ** (steps / 5e4)

# main class

class Trainer:
    """ Section 5.4 """

    @typecheck
    def __init__(
        self,
        model: Alphafold3,
        *,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int = 1,
        map_dataset_input_fn: Callable | None = None,
        valid_dataset: Dataset | None = None,
        valid_every: int = 1000,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema_decay = 0.999,
        lr = 1.8e-3,
        default_adam_kwargs: dict = dict(
            betas = (0.9, 0.95),
            eps = 1e-8
        ),
        clip_grad_norm = 10.,
        default_lambda_lr = default_lambda_lr_fn,
        fabric: Fabric | None = None,
        accelerator = 'auto',
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        overwrite_checkpoints: bool = False,
        fabric_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()

        if not exists(fabric):
            fabric = Fabric(accelerator = accelerator, **fabric_kwargs)

        self.fabric = fabric
        fabric.launch()

        # model

        self.model = model

        # exponential moving average

        if self.is_main:
            self.ema_model = EMA(
                model,
                beta = ema_decay,
                include_online_model = False,
                **ema_kwargs
            )

        # optimizer

        if not exists(optimizer):
            optimizer = Adam(
                model.parameters(),
                lr = lr,
                **default_adam_kwargs
            )

        self.optimizer = optimizer

        # if map dataset function given, curry into DataLoader

        DataLoader_ = DataLoader

        if exists(map_dataset_input_fn):
            DataLoader_ = partial(DataLoader_, map_input_fn = map_dataset_input_fn)

        # train dataloader

        self.dataloader = DataLoader_(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        # validation dataloader on the EMA model

        self.valid_every = valid_every

        self.needs_valid = exists(valid_dataset)

        if self.needs_valid and self.is_main:
            self.valid_dataset_size = len(valid_dataset)
            self.valid_dataloader = DataLoader_(valid_dataset, batch_size = batch_size)

        # testing dataloader on EMA model

        self.needs_test = exists(test_dataset)

        if self.needs_test and self.is_main:
            self.test_dataset_size = len(test_dataset)
            self.test_dataloader = DataLoader_(test_dataset, batch_size = batch_size)

        # training steps and num gradient accum steps

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        # setup fabric

        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        fabric.setup_dataloaders(self.dataloader)

        # scheduler

        if not exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda = default_lambda_lr)

        self.scheduler = scheduler

        # gradient clipping norm

        self.clip_grad_norm = clip_grad_norm

        # steps

        self.steps = 0

        # checkpointing logic

        self.checkpoint_every = checkpoint_every
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

    @property
    def is_main(self):
        return self.fabric.global_rank == 0

    # saving and loading

    def save(self, path: str | Path, overwrite = False):
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok = True, parents = True)

        package = dict(
            model = self.model.state_dict_with_init_args,
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            steps = self.steps
        )

        torch.save(package, str(path))

    def load(self, path: str | Path, strict = True):
        if isinstance(path, str):
            path = Path(path)

        assert path.exists()

        self.model.load(path)

        package = torch.load(str(path))

        if 'optimizer' in package:
            self.optimizer.load_state_dict(package['optimizer'])

        if 'scheduler' in package:
            self.scheduler.load_state_dict(package['scheduler'])

        self.steps = package.get('steps', 0)

    # shortcut methods

    def wait(self):
        self.fabric.barrier()

    def print(self, *args, **kwargs):
        self.fabric.print(*args, **kwargs)

    def log(self, **log_data):
        self.fabric.log_dict(log_data, step = self.steps)

    # main train forwards

    def __call__(
        self
    ):
        dl = cycle(self.dataloader)

        # while less than required number of training steps

        while self.steps < self.num_train_steps:

            self.model.train()

            # gradient accumulation

            total_loss = 0.
            train_loss_breakdown = None

            for grad_accum_step in range(self.grad_accum_every):
                is_accumulating = grad_accum_step < (self.grad_accum_every - 1)

                inputs = next(dl)

                with self.fabric.no_backward_sync(self.model, enabled = is_accumulating):

                    # model forwards

                    loss, loss_breakdown = self.model(
                        **inputs,
                        return_loss_breakdown = True
                    )

                    # accumulate

                    scale = self.grad_accum_every ** -1

                    total_loss += loss.item() * scale
                    train_loss_breakdown = accum_dict(train_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                    # backwards

                    self.fabric.backward(loss / self.grad_accum_every)

            # log entire loss breakdown

            self.log(**train_loss_breakdown)

            self.print(f'loss: {total_loss:.3f}')

            # clip gradients

            self.fabric.clip_gradients(self.model, self.optimizer, max_norm = self.clip_grad_norm)

            # optimizer step

            self.optimizer.step()

            # update exponential moving average

            self.wait()

            if self.is_main:
                self.ema_model.update()

            self.wait()

            # scheduler

            self.scheduler.step()
            self.optimizer.zero_grad()

            self.steps += 1

            # maybe validate, for now, only on main with EMA model

            if (
                self.is_main and
                self.needs_valid and
                divisible_by(self.steps, self.valid_every)
            ):
                with torch.no_grad():
                    self.ema_model.eval()

                    total_valid_loss = 0.
                    valid_loss_breakdown = None

                    for valid_batch in self.valid_dataloader:
                        valid_loss, loss_breakdown = self.ema_model(
                            **valid_batch,
                            return_loss_breakdown = True
                        )

                        valid_batch_size = valid_batch.get('atom_inputs').shape[0]
                        scale = valid_batch_size / self.valid_dataset_size

                        total_valid_loss += valid_loss.item() * scale
                        valid_loss_breakdown = accum_dict(valid_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                    self.print(f'valid loss: {total_valid_loss:.3f}')

                # prepend valid_ to all losses for logging

                valid_loss_breakdown = {f'valid_{k}':v for k, v in valid_loss_breakdown.items()}

                # log

                self.log(**valid_loss_breakdown)

            self.wait()

            if self.is_main and divisible_by(self.steps, self.checkpoint_every):
                checkpoint_path = self.checkpoint_folder / f'af3.ckpt.{self.steps}.pt'

                self.save(checkpoint_path, overwrite = self.overwrite_checkpoints)

            self.wait()

        # maybe test

        if self.is_main and self.needs_test:
            with torch.no_grad():
                self.ema_model.eval()

                total_test_loss = 0.
                test_loss_breakdown = None

                for test_batch in self.test_dataloader:
                    test_loss, loss_breakdown = self.ema_model(
                        **test_batch,
                        return_loss_breakdown = True
                    )

                    test_batch_size = test_batch.get('atom_inputs').shape[0]
                    scale = test_batch_size / self.test_dataset_size

                    total_test_loss += test_loss.item() * scale
                    test_loss_breakdown = accum_dict(test_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                self.print(f'test loss: {total_test_loss:.3f}')

            # prepend test_ to all losses for logging

            test_loss_breakdown = {f'test_{k}':v for k, v in test_loss_breakdown.items()}

            # log

            self.log(**test_loss_breakdown)

        print('training complete')
