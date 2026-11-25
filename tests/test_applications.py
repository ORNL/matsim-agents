"""Example applications: they load, they are self-consistent, and the fusion study is pinned.

The framework hardcodes no material system. These tests check that each application under
``examples/`` supplies its own through configuration, and that the 1,024-site fusion study
keeps its exact requirements.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from matsim_agents.core.application import (
    ApplicationConfig,
    SupercellTarget,
    load_application,
)
from matsim_agents.workflows.full_supercell_campaign import FullSupercellCampaign
from matsim_agents.workflows.hierarchy.config import load_campaign_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FUSION = EXAMPLES / "fusion_bcc_1024" / "application.yaml"
CANTOR = EXAMPLES / "cantor_fcc_256" / "application.yaml"
PEROVSKITE = EXAMPLES / "perovskite_oxides" / "application.yaml"


def test_fusion_1024_requirements_are_preserved_exactly():
    app = load_application(FUSION)
    t = app.target
    assert t.composition == {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}
    assert list(t.composition) == ["W", "Ta", "Cr", "V", "Hf", "Mo"]
    assert t.formula == "W640Ta128Cr64V96Hf32Mo64"
    assert t.supercell == (8, 8, 8)
    assert t.parent_lattice == "bcc"
    assert t.expected_sites == t.n_sites == 1024
    assert app.workflow == "full_supercell_campaign"


def test_fusion_campaign_builds_the_application_target():
    app = load_application(FUSION)
    campaign = FullSupercellCampaign.from_yaml(app.resolve(app.workflow_config))
    parent = campaign.parent_supercell
    assert parent.lattice == app.target.parent_lattice
    assert parent.conventional_repeats == app.target.supercell
    assert parent.expected_parent_atoms == app.target.expected_sites
    assert parent.composition == app.target.composition
    assert parent.lattice_parameter == app.parameters["lattice_parameter_angstrom"]
    assert campaign.numerical_qualification.require_full_parent_supercell


def test_cantor_fcc_is_a_non_bcc_hierarchy_campaign():
    app = load_application(CANTOR)
    assert app.target.parent_lattice == "fcc"
    assert app.target.n_sites == 256
    cfg = load_campaign_config(app.resolve(app.workflow_config))
    assert cfg.target.model_dump() == {**app.target.model_dump(), "reference": ""}
    assert cfg.elements == ("Co", "Cr", "Fe", "Mn", "Ni")
    assert cfg.orders == (2, 3, 4, 5)
    assert cfg.execution.dry_run


def test_cantor_campaign_plans_offline(tmp_path):
    from matsim_agents.workflows.hierarchy.engine import CampaignWorkspace, inventory

    app = load_application(CANTOR)
    cfg = load_campaign_config(app.resolve(app.workflow_config))
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"candidates": None})})
    report = inventory(CampaignWorkspace(tmp_path / "ws", cfg))
    assert report is not None


def test_perovskite_is_a_composition_driven_paradigm(tmp_path):
    app = load_application(PEROVSKITE)
    assert app.target is None
    assert app.paradigm != load_application(FUSION).paradigm
    assert app.workflow == "composition_exploration"
    assert "SrTiO3" in app.compositions


def test_perovskite_runner_generates_prototype_candidates(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("perovskite_run", PEROVSKITE.parent / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = module.main(["-o", str(tmp_path)])
    assert set(counts) == set(load_application(PEROVSKITE).compositions)
    assert all(n > 0 for n in counts.values())


def test_target_rejects_inconsistent_site_counts():
    with pytest.raises(ValidationError, match="1024 sites"):
        SupercellTarget(
            composition={"W": 1024},
            supercell=(8, 8, 4),
            parent_lattice="bcc",
            expected_sites=1024,
        )
    with pytest.raises(ValidationError, match="sum to 256"):
        SupercellTarget(
            composition={"Ni": 255},
            supercell=(4, 4, 4),
            parent_lattice="fcc",
            expected_sites=256,
        )


def test_application_needs_a_material_system_and_matching_species():
    with pytest.raises(ValidationError):
        ApplicationConfig(name="empty", paradigm="x", workflow="y")
    with pytest.raises(ValidationError):
        ApplicationConfig(
            name="bad-species",
            paradigm="supercell_alloy",
            workflow="hierarchy",
            target={
                "composition": {"Ni": 4},
                "supercell": (1, 1, 1),
                "parent_lattice": "fcc",
                "expected_sites": 4,
            },
            species=("Cu",),
        )


def test_application_with_missing_file_is_rejected(tmp_path):
    path = tmp_path / "application.yaml"
    path.write_text(
        "name: missing\nparadigm: prototype_search\nworkflow: composition_exploration\n"
        "compositions: [NaCl]\nworkflow_config: nowhere.yaml\n"
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        load_application(path)
