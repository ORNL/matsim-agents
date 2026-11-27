"""The surrogate model's calculators and MD worker pool across iterations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from matsim_agents.backends.mlip.calculator import build_ensemble, make_mlip_calculator
from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.backends.mlip.pool import (
    MLIPWorkerPool,
    release_device_memory,
)
from matsim_agents.execution.placement import Placement


class _SurrogateModels:
    """The surrogate's calculators and MD worker pool across AL iterations.

    When no other pool uses the mlip devices (``keep``), the calculators and the
    MD workers stay loaded between iterations and are rebuilt only when the
    model changes (a promotion). Otherwise they are built each iteration and
    released before DFT and training, which use the same devices.
    """

    def __init__(self, placement: Placement, *, keep: bool) -> None:
        self.placement = placement
        self.keep = keep
        self._key: tuple[str, bool] | None = None
        self._primary: Any = None
        self._ensemble: list = []
        self._pool: MLIPWorkerPool | None = None

    def calculators(self, mlip_cfg: MLIPConfig, enable_drop: bool) -> tuple[Any, list]:
        key = (mlip_cfg.model_dump_json(), enable_drop)
        if key != self._key:
            self._drop()
            self._primary = make_mlip_calculator(mlip_cfg, enable_mc_dropout=enable_drop)
            self._ensemble = (
                build_ensemble(mlip_cfg, enable_mc_dropout=enable_drop)
                if mlip_cfg.ensemble_models
                else []
            )
            self._key = key
        return self._primary, self._ensemble

    @contextmanager
    def md_pool(self) -> Iterator[MLIPWorkerPool]:
        """Workers running the MD seeds concurrently, one per mlip device."""
        if not self.keep:
            # The workers exit afterwards so their memory is free for DFT.
            with MLIPWorkerPool.from_placement(self.placement) as pool:
                yield pool
            return
        if self._pool is None:
            self._pool = MLIPWorkerPool.from_placement(self.placement)
        yield self._pool

    def after_scoring(self) -> None:
        """Free the mlip devices for DFT and training unless the models stay loaded."""
        if not self.keep:
            self._drop()

    def incumbent_calculator(self, mlip_cfg: MLIPConfig) -> Any:
        """The loaded deterministic incumbent model, for the promotion comparison."""
        if self.keep and self._key == (mlip_cfg.model_dump_json(), False):
            return self._primary
        return None

    def _drop(self) -> None:
        if self._key is not None:
            self._primary, self._ensemble, self._key = None, [], None
            release_device_memory()

    def close(self) -> None:
        self._drop()
        if self._pool is not None:
            self._pool.close()
            self._pool = None
