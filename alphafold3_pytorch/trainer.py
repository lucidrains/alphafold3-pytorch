from __future__ import annotations

from pathlib import Path

from alphafold3_pytorch.alphafold3 import Alphafold3

from typing import TypedDict
from alphafold3_pytorch.typing import (
    typecheck,
    Int, Bool, Float
)

import torch
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from ema_pytorch import EMA

from lightning import Fabric

# constants

@typecheck
class Alphafold3Input(TypedDict):
    atom_inputs:                Float['m dai']
    residue_atom_lens:          Int['n 2']
    atompair_inputs:            Float['m m dap']
    additional_residue_feats:   Float['n 10']
    templates:                  Float['t n n dt']
    template_mask:              Bool['t'] | None
    msa:                        Float['s n dm']
    msa_mask:                   Bool['s'] | None
    atom_pos:                   Float['m 3'] | None
    residue_atom_indices:       Int['n'] | None
    distance_labels:            Int['n n'] | None
    pae_labels:                 Int['n n'] | None
    pde_labels:                 Int['n'] | None
    resolved_labels:            Int['n'] | None

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

        # train dataloader

        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        # validation dataloader on the EMA model

        self.valid_every = valid_every

        self.needs_valid = exists(valid_dataset)

        if self.needs_valid and self.is_main:
            self.valid_dataset_size = len(valid_dataset)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size)

        # testing dataloader on EMA model

        self.needs_test = exists(test_dataset)

        if self.needs_test and self.is_main:
            self.test_dataset_size = len(test_dataset)
            self.test_dataloader = DataLoader(test_dataset, batch_size = batch_size)

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

        print(f'training complete')
