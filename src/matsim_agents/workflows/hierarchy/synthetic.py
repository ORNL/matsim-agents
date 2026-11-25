"""Synthetic hierarchical datasets for software tests of the transfer machinery.

These contain no physics. Passing tests on them shows the machinery can detect useful transfer,
negative transfer and missing evidence, not that any real alloy transfers.
"""

from __future__ import annotations

import itertools

import numpy as np
from pydantic import BaseModel, Field

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.workflows.candidate_registry import chemical_system
from matsim_agents.workflows.hierarchy.transfer import TransferData


def feature_vector(fractions: np.ndarray, sro: float) -> np.ndarray:
    """Singles, pair products, triple products and one SRO scalar (42 features for 6 elements)."""
    n = len(fractions)
    pairs = [fractions[i] * fractions[j] for i, j in itertools.combinations(range(n), 2)]
    triples = [
        fractions[i] * fractions[j] * fractions[k]
        for i, j, k in itertools.combinations(range(n), 3)
    ]
    return np.concatenate([fractions, pairs, triples, [sro]])


class SyntheticSpec(BaseModel):
    seed: int = 0
    property: str = "bulk"
    elements: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    magnetic_element: str | None = "C"
    orders: tuple[int, ...] = (2, 3, 4, 5, 6)
    compositions_per_system: int = Field(6, ge=1)
    n_orderings: int = Field(4, ge=2)
    mismatched_orders: tuple[int, ...] = ()
    mismatch: float = Field(0.0, ge=0.0)
    higher_order_strength: float = Field(1.0, ge=0.0)
    noise: float = Field(0.01, ge=0.0)
    cost_per_order: float = Field(0.25, gt=0.0)


def make_synthetic_hierarchy(spec: SyntheticSpec | None = None) -> TransferData:
    """Rows over every element subset of each order; mismatched orders use other pair terms."""
    spec = spec or SyntheticSpec()
    elements = tuple(sorted(spec.elements))
    n_el = len(elements)
    truth = numpy_rng(spec.seed, "synthetic-truth", spec.property)
    single = truth.normal(0, 1, n_el)
    pair = 2.0 * truth.normal(0, 1, n_el * (n_el - 1) // 2)
    triple = spec.higher_order_strength * truth.normal(0, 1, n_el * (n_el - 1) * (n_el - 2) // 6)
    sro_weight = truth.normal(0, 1)
    shift_single = spec.mismatch * truth.normal(0, 1, n_el)
    shift_pair = spec.mismatch * truth.normal(0, 1, len(pair))
    rows = numpy_rng(spec.seed, "synthetic-rows")
    noise = numpy_rng(spec.seed, "synthetic-noise", spec.property)
    ordering_sro = np.linspace(-1.0, 1.0, spec.n_orderings)
    mag = elements.index(spec.magnetic_element) if spec.magnetic_element in elements else None

    columns: dict[str, list] = {
        k: []
        for k in (
            "ids",
            "X",
            "y",
            "system",
            "composition_id",
            "ordering_id",
            "environment",
            "sro_regime",
            "magnetic",
            "cost",
        )
    }
    for order in spec.orders:
        for subset in itertools.combinations(range(n_el), order):
            system = chemical_system(elements[i] for i in subset)
            for _ in range(spec.compositions_per_system):
                fractions = np.zeros(n_el)
                fractions[list(subset)] = rows.dirichlet(np.full(order, 2.0))
                comp_id = f"{system}:" + ",".join(f"{x:.4f}" for x in fractions)
                for o in range(spec.n_orderings):
                    sro = float(ordering_sro[o] + 0.05 * rows.normal())
                    x = feature_vector(fractions, sro)
                    bad = order in spec.mismatched_orders
                    w = np.concatenate(
                        [
                            single + (shift_single if bad else 0.0),
                            pair + (shift_pair if bad else 0.0),
                            triple,
                            [sro_weight],
                        ]
                    )
                    y = float(x @ w + spec.noise * noise.normal())
                    columns["ids"].append(f"{spec.property}-{len(columns['ids'])}")
                    columns["X"].append(x)
                    columns["y"].append(y)
                    columns["system"].append(system)
                    columns["composition_id"].append(comp_id)
                    columns["ordering_id"].append(f"ord{o}")
                    columns["environment"].append(elements[int(np.argmax(fractions))])
                    strength = abs(ordering_sro[o])
                    columns["sro_regime"].append(
                        "random" if strength < 0.2 else "weak" if strength < 0.6 else "strong"
                    )
                    magnetic = mag is not None and fractions[mag] > 0.25
                    columns["magnetic"].append("fm" if magnetic else "nm")
                    columns["cost"].append(spec.cost_per_order * order)
    return TransferData(
        property=spec.property,
        ids=tuple(columns["ids"]),
        X=np.asarray(columns["X"]),
        y=np.asarray(columns["y"]),
        system=tuple(columns["system"]),
        composition_id=tuple(columns["composition_id"]),
        ordering_id=tuple(columns["ordering_id"]),
        environment=tuple(columns["environment"]),
        sro_regime=tuple(columns["sro_regime"]),
        magnetic=tuple(columns["magnetic"]),
        cost_node_hours=np.asarray(columns["cost"]),
    )
