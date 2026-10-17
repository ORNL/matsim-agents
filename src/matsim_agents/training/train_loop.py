"""Epoch loop shared by the HydraGNN fine-tuners: train, validate, keep the best weights."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


def train_keep_best(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn: Callable[[Any, Any], Any],
    *,
    dev: Any,
    epochs: int,
    n_train: int,
    n_val: int,
    train_mode: bool,
    scheduler=None,
    grad_clip: float = 0.0,
) -> None:
    """Train for ``epochs`` and restore the weights with the lowest validation loss.

    On these small AL datasets the loss overfits well before the last epoch, so
    the best-validation checkpoint is kept rather than the final one. With
    ``train_mode=False`` the model stays in ``eval()`` during training, which
    keeps a frozen backbone's normalisation statistics fixed.
    """
    import torch

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(epochs):
        model.train() if train_mode else model.eval()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.detach()) * batch.num_graphs
        train_mse = running / max(n_train, 1)
        val_msg = ""
        selection_mse = train_mse
        if val_loader is not None:
            model.eval()
            vrun = 0.0
            for batch in val_loader:
                batch = batch.to(dev)
                vloss = loss_fn(model, batch)
                vrun += float(vloss.detach()) * batch.num_graphs
            selection_mse = vrun / max(n_val, 1)
            val_msg = f" val={selection_mse:.6f}"
        log.info("Epoch %d/%d: train=%.6f%s", epoch + 1, epochs, train_mse, val_msg)
        if selection_mse < best_val:
            best_val = selection_mse
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Best epoch %d/%d: val=%.6f (restored for save)", best_epoch, epochs, best_val)
