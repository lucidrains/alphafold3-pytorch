from typing import Tuple, TypedDict

import torch
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric, MinMetric

from alphafold3_pytorch.utils import RankedLogger
from alphafold3_pytorch.utils.model_utils import default_lambda_lr_fn
from alphafold3_pytorch.utils.typing import Bool, Float, Int, typecheck

# constants


@typecheck
class AlphaFold3Input(TypedDict):
    """A collection of inputs to AlphaFold 3."""

    atom_inputs: Float["m dai"]  # type: ignore
    residue_atom_lens: Int["n 2"]  # type: ignore
    atompair_feats: Float["m m dap"]  # type: ignore
    additional_residue_feats: Float["n 10"]  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    template_mask: Bool["t"] | None  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    msa_mask: Bool["s"] | None  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    residue_atom_indices: Int["n"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int["n"] | None  # type: ignore
    resolved_labels: Int["n"] | None  # type: ignore


AVAILABLE_LR_SCHEDULERS = ["wcd", "plateau"]

log = RankedLogger(__name__, rank_zero_only=False)

# lightning module


class AlphaFold3LitModule(LightningModule):
    """A `LightningModule` for AlphaFold 3.
    Implements details from Section 5.4 of the paper.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

    @property
    def is_main(self):
        return self.trainer.global_rank == 0

    def forward(self, batch: AlphaFold3Input) -> Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A batch of `AlphaFold3Input` data.
        :return: A tensor of losses.
        """
        return self.net(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: AlphaFold3Input) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of `AlphaFold3Input` data.
        :return: A tensor of losses.
        """
        loss = self.forward(batch)
        return loss

    def training_step(self, batch: AlphaFold3Input, batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of `AlphaFold3Input` data.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: AlphaFold3Input, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of `AlphaFold3Input` data.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(val_loss)  # update best so far val loss
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: AlphaFold3Input, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of `AlphaFold3Input` data.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def on_after_backward(self):
        """Skip updates in case of unstable gradients.

        Reference: https://github.com/Lightning-AI/lightning/issues/4956
        """
        if self.hparams.skip_invalid_gradient_updates:
            valid_gradients = True
            for _, param in self.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (
                        torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                    )
                    if not valid_gradients:
                        break
            if not valid_gradients:
                log.warning(
                    f"Detected `inf` or `nan` values in gradients at global step {self.trainer.global_step}. Not updating model parameters."
                )
                self.zero_grad()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and optional learning-rate schedulers to use during model
        optimization.

        :return: Configured optimizer(s) and optional learning-rate scheduler(s) to be used for
            training.
        """
        try:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        except TypeError:
            # NOTE: Trainer strategies such as DeepSpeed require `params` to instead be specified as `model_params`
            optimizer = self.hparams.optimizer(model_params=self.trainer.model.parameters())
        if self.hparams.scheduler is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=default_lambda_lr_fn, verbose=True
            )
        else:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "step",
                "frequency": 1,
                "name": "lambda_lr",
            },
        }


if __name__ == "__main__":
    _ = AlphaFold3LitModule(None, None, None, None)
