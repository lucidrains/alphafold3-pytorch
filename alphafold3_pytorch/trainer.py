from __future__ import annotations

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
    atompair_feats:             Float['m m dap']
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

def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

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

        # data

        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

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

    @property
    def is_main(self):
        return self.fabric.global_rank == 0

    def __call__(
        self
    ):
        dl = iter(self.dataloader)

        steps = 0

        while steps < self.num_train_steps:

            for grad_accum_step in range(self.grad_accum_every):
                is_accumulating = grad_accum_step < (self.grad_accum_every - 1)

                inputs = next(dl)

                with self.fabric.no_backward_sync(self.model, enabled = is_accumulating):
                    loss = self.model(**inputs)

                self.fabric.backward(loss / self.grad_accum_every)

            print(f'loss: {loss.item():.3f}')

            self.fabric.clip_gradients(self.model, self.optimizer, max_norm = self.clip_grad_norm)

            self.optimizer.step()

            if self.is_main:
                self.ema_model.update()

            self.scheduler.step()
            self.optimizer.zero_grad()

            steps += 1

        print(f'training complete')
