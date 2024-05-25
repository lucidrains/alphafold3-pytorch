from __future__ import annotations

from alphafold3_pytorch.alphafold3 import Alphafold3
from alphafold3_pytorch.typing import typecheck

import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from ema_pytorch import EMA

from lightning import Fabric

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

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

        # setup fabric

        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        # scheduler

        if not exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda = default_lambda_lr)

        self.scheduler = scheduler

        # gradient clipping norm

        self.clip_grad_norm = clip_grad_norm

    def __call__(
        self
    ):
        pass
