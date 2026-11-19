"""Central seed management: stream derivation and the repository-wide policy.

Every random generator comes from :mod:`matsim_agents.execution.seeding`, and
seeds are configured values: no generator is built elsewhere, no seed is a
literal or ``seed + i`` arithmetic, and no seed parameter has a default.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import pytest

from matsim_agents.execution.seeding import (
    numpy_rng,
    python_rng,
    seed_process,
    seeded_torch,
    stream_seed,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDING = ROOT / "src" / "matsim_agents" / "execution" / "seeding.py"
CHECKED = [
    *sorted((ROOT / "src" / "matsim_agents").rglob("*.py")),
    *sorted((ROOT / "deployments").rglob("*.py")),
]
GENERATORS = {
    "default_rng",
    "SeedSequence",
    "RandomState",
    "Random",
    "Generator",
    "manual_seed",
    "manual_seed_all",
    "seed",
}


def test_bare_non_negative_seed_is_the_root_stream():
    assert stream_seed(20261113) == 20261113
    assert numpy_rng(7).random(3).tolist() == np.random.default_rng(7).random(3).tolist()
    assert python_rng(7).random() == random.Random(7).random()


def test_streams_are_deterministic_distinct_and_32_bit():
    seeds = {stream_seed(5, "md", index) for index in range(1000)}
    assert len(seeds) == 1000
    assert all(0 <= seed < 2**32 for seed in seeds)
    assert stream_seed(5, "md", 3) == stream_seed(5, "md", 3)
    assert stream_seed(5, "md", 3) != stream_seed(5, "acquisition", 3)
    assert stream_seed(5, "md", 1) != stream_seed(6, "md", 0)
    assert stream_seed(5, "1") != stream_seed(5, 1)


def test_negative_seeds_do_not_alias_positive_ones():
    assert stream_seed(-3) != stream_seed(3)
    assert python_rng(-3).random() != python_rng(3).random()


@pytest.mark.parametrize("bad", [1.0, "1", True, None])
def test_seeds_must_be_integers(bad):
    with pytest.raises(TypeError):
        stream_seed(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1.5, True, None, ("md",)])
def test_stream_keys_must_be_strings_or_integers(bad):
    with pytest.raises(TypeError):
        stream_seed(1, bad)  # type: ignore[arg-type]


def test_seeded_torch_reproduces_and_restores_the_global_generator():
    torch = pytest.importorskip("torch")
    torch.manual_seed(123)
    expected_next = torch.rand(2)
    torch.manual_seed(123)
    with seeded_torch(9, "mc-dropout"):
        first = torch.rand(4)
    with seeded_torch(9, "mc-dropout"):
        assert torch.equal(torch.rand(4), first)
    assert torch.equal(torch.rand(2), expected_next)


def test_seed_process_seeds_every_global_generator():
    seed_process(11, "training")
    draws = (random.random(), np.random.random())
    seed_process(11, "training")
    assert (random.random(), np.random.random()) == draws


def _seed_name(name: str | None) -> bool:
    return bool(name) and (name == "seed" or name.endswith("_seed"))


def _unconfigured(value: ast.expr) -> bool:
    """A literal integer seed or a seed computed by arithmetic."""
    return isinstance(value, ast.BinOp) or (
        isinstance(value, ast.Constant) and isinstance(value.value, int)
    )


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    def report(node: ast.AST, message: str) -> None:
        found.append(f"{path.relative_to(ROOT)}:{node.lineno}: {message}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            owner = ast.unparse(func.value) if isinstance(func, ast.Attribute) else ""
            random_module = owner in {"random", "np.random", "numpy.random", "torch", "torch.cuda"}
            if path != SEEDING and (
                (name in GENERATORS and (random_module or name in {"default_rng", "SeedSequence"}))
                or (owner in {"random", "np.random", "numpy.random"} and name != "Generator")
            ):
                report(node, f"random generator {ast.unparse(func)}() outside execution/seeding")
            for keyword in node.keywords:
                if _seed_name(keyword.arg) and _unconfigured(keyword.value):
                    report(
                        node, f"{keyword.arg}={ast.unparse(keyword.value)} is not a configured seed"
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module in {"random", "numpy.random", "torch.random"}
            and path != SEEDING
            and any(alias.name != "Generator" for alias in node.names)
        ):
            report(node, f"from {node.module} import bypasses execution/seeding")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _seed_name(key.value)
                    and isinstance(value, ast.BinOp)
                ):
                    report(
                        node, f"{key.value!r}: {ast.unparse(value)} derives a seed by arithmetic"
                    )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            defaults = [
                *zip(
                    positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
                ),
                *(
                    (arg, default)
                    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
                    if default
                ),
            ]
            for arg, default in defaults:
                if _seed_name(arg.arg) and _unconfigured(default):
                    report(node, f"{node.name}({arg.arg}={ast.unparse(default)}) defaults a seed")
    return found


def test_no_random_generator_or_seed_outside_the_seeding_module():
    violations = [violation for path in CHECKED for violation in _violations(path)]
    assert not violations, "\n".join(violations)
