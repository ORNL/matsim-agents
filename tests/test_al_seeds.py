"""Unit tests for active-learning seed-source resolution.

Covers:
* `kind: paths` — round-trip with on-disk files; missing-file error.
* `kind: prompt` — the LLM is replaced by a stub that returns canned JSON,
  and we verify the LLM proposal is persisted to disk for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from matsim_agents.active_learning.config import LLMSeedConfig, SeedSourceConfig
from matsim_agents.active_learning.seeds import resolve_seed_structures


def _write_dummy_structure(p: Path) -> None:
    """Write a minimal, ASE-readable POSCAR (we never actually parse it here)."""
    p.write_text(
        "Si\n"
        "1.0\n"
        "2.715  2.715  0.000\n"
        "0.000  2.715  2.715\n"
        "2.715  0.000  2.715\n"
        "Si\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n"
    )


# --------------------------------------------------------------------------- #
# kind: paths                                                                 #
# --------------------------------------------------------------------------- #


def test_paths_round_trip(tmp_path: Path) -> None:
    a = tmp_path / "a.vasp"
    b = tmp_path / "b.vasp"
    _write_dummy_structure(a)
    _write_dummy_structure(b)

    cfg = SeedSourceConfig(kind="paths", paths=[a, b])
    out = resolve_seed_structures(cfg, tmp_path / "seeds_out")
    assert [p.name for p in out] == ["a.vasp", "b.vasp"]


def test_paths_missing_file_raises(tmp_path: Path) -> None:
    a = tmp_path / "exists.vasp"
    _write_dummy_structure(a)
    cfg = SeedSourceConfig(kind="paths", paths=[a, tmp_path / "ghost.vasp"])
    with pytest.raises(FileNotFoundError):
        resolve_seed_structures(cfg, tmp_path / "seeds_out")


# --------------------------------------------------------------------------- #
# kind: prompt — uses a stub LLM                                              #
# --------------------------------------------------------------------------- #


@dataclass
class _StubResp:
    content: str


@dataclass
class _StubLLM:
    """Minimal stand-in for a langchain ChatModel with `.invoke([...])`."""

    canned: str

    def invoke(self, _messages):  # noqa: ANN001
        return _StubResp(content=self.canned)


def _patch_llm_and_phases(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_response: str,
    written_files: list[Path],
) -> None:
    """Patch both the LLM factory and the phase enumerator used by seeds.py."""

    def _fake_get_chat_model(**_kwargs):  # noqa: ANN001, ANN003
        return _StubLLM(canned=llm_response)

    monkeypatch.setattr(
        "matsim_agents.llm.get_chat_model",
        _fake_get_chat_model,
    )

    # Stub the prototype enumerator so the test does not depend on
    # pymatgen / discovery machinery at all. It just creates one VASP
    # file per requested formula and returns synthetic candidates.
    @dataclass
    class _PhaseCand:
        structure_path: str

    def _fake_parse_composition(formula: str):
        @dataclass
        class _C:
            formula: str

        return _C(formula=formula)

    def _fake_enumerate_phases(comp, out_dir, **_kwargs):  # noqa: ANN001, ANN003
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{comp.formula}_proto.vasp"
        _write_dummy_structure(path)
        written_files.append(path)
        return [_PhaseCand(structure_path=str(path))]

    monkeypatch.setattr(
        "matsim_agents.discovery.parse_composition",
        _fake_parse_composition,
    )
    monkeypatch.setattr(
        "matsim_agents.discovery.enumerate_phases",
        _fake_enumerate_phases,
    )


def test_prompt_uses_llm_and_persists_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[Path] = []
    canned = json.dumps({"compositions": ["Si", "Ge"]})
    _patch_llm_and_phases(monkeypatch, llm_response=canned, written_files=written)

    cfg = SeedSourceConfig(
        kind="prompt",
        prompt="suggest two simple group-IV elements",
        llm=LLMSeedConfig(provider="ollama", model="qwen2.5:14b"),
        max_compositions=2,
    )
    out_dir = tmp_path / "seeds"
    seeds = resolve_seed_structures(cfg, out_dir)

    # Every formula yielded one fake prototype seed.
    assert len(seeds) == 2
    assert {p.name for p in seeds} == {"Si_proto.vasp", "Ge_proto.vasp"}

    # The LLM proposal must be persisted for reproducibility.
    proposal = json.loads((out_dir / "llm_proposed_compositions.json").read_text())
    assert proposal["compositions"] == ["Si", "Ge"]
    assert proposal["prompt"].startswith("suggest")


def test_prompt_with_unparseable_llm_output_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[Path] = []
    _patch_llm_and_phases(monkeypatch, llm_response="sorry, I can't help", written_files=written)
    cfg = SeedSourceConfig(
        kind="prompt",
        prompt="anything",
        llm=LLMSeedConfig(),
    )
    with pytest.raises(RuntimeError, match=r"did not return any usable compositions"):
        resolve_seed_structures(cfg, tmp_path / "seeds")


def test_prompt_strips_markdown_code_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser must tolerate the LLM wrapping JSON in ```json … ``` fences."""
    written: list[Path] = []
    canned = '```json\n{"compositions": ["Si"]}\n```'
    _patch_llm_and_phases(monkeypatch, llm_response=canned, written_files=written)

    cfg = SeedSourceConfig(
        kind="prompt",
        prompt="just one element please",
        llm=LLMSeedConfig(),
        max_compositions=1,
    )
    seeds = resolve_seed_structures(cfg, tmp_path / "seeds")
    assert [p.name for p in seeds] == ["Si_proto.vasp"]
