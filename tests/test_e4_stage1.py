from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.aurora_dft.planning import build_candidate_inventory  # noqa: E402
from benchmarks.e4.aurora_dft.prepare_stage1 import (  # noqa: E402
    _bcc_stage_incar,
    _bcc_stage_records,
    _fcc_parent,
    _fcc_stage_records,
    _lifepo4_antisites,
    _lifepo4_defect_metadata,
    _stage_pbs,
    _write_poscar,
)


def test_bcc_production_incars_use_approved_force_threshold_without_changing_start():
    archived_start = (
        "ISTART = 0\nICHARG = 2\nPREC = PRECISION\n"
        "EDIFF = 1.0E-6\nNSW = 240\nIBRION = 2\nISIF = 3\n"
    )

    low = _bcc_stage_incar(archived_start, "Low", restart=False)
    normal = _bcc_stage_incar(archived_start, "Normal", restart=True)

    assert "EDIFF = 1.0E-6\n" in low and "EDIFF = 1.0E-6\n" in normal
    assert "NSW = 240\n" in low and "NSW = 240\n" in normal
    assert low.count("EDIFFG = -0.01\n") == 1
    assert normal.count("EDIFFG = -0.01\n") == 1
    assert "PREC = Low" in low and "PREC = Normal" in normal
    assert "ISTART = 0\nICHARG = 2" in low
    assert "ISTART = 1\nICHARG = 1" in normal
    assert "EDIFFG" not in archived_start


def test_stage1_selection_covers_all_bcc_orders_and_four_fcc_compositions():
    candidates = build_candidate_inventory()
    bcc = _bcc_stage_records(candidates)
    fcc = _fcc_stage_records(candidates)
    assert len(bcc) == 8
    assert {len(row["elements"]) for row in bcc} == {4, 5, 6}
    assert len({tuple(row["elements"]) for row in bcc if len(row["elements"]) == 4}) == 3
    assert len({tuple(row["elements"]) for row in bcc if len(row["elements"]) == 5}) == 3
    assert len(fcc) == 4
    assert len({tuple(row["composition"].items()) for row in fcc}) == 4
    lifepo4 = [row for row in candidates if row["material_family"] == "lifepo4"]
    assert len(lifepo4) == 510
    assert {
        count: sum(row["antisite_exchange_pairs"] == count for row in lifepo4) for count in range(4)
    } == {0: 1, 1: 8, 2: 496, 3: 5}


def test_fcc_stage1_supercell_is_128_atoms():
    parent_108 = bulk("Ni", "fcc", a=3.576, cubic=True).repeat((3, 3, 3))
    stage1 = _fcc_parent(parent_108)
    assert len(stage1) == 128
    assert np.allclose(stage1.cell.lengths(), [14.304, 14.304, 7.152])


def test_lifepo4_inventory_is_capped_and_symmetry_unique():
    symbols = ["Li"] * 16 + ["Fe"] * 16 + ["P"] * 16 + ["O"] * 64
    parent = Atoms(
        symbols,
        scaled_positions=np.random.default_rng(42).random((112, 3)),
        cell=np.eye(3) * 20,
        pbc=True,
    )
    identity_permutation = [list(range(len(parent)))]

    candidates = _lifepo4_antisites(parent, identity_permutation, limit=510)

    assert len(candidates) == 510
    assert len({canonical for canonical, _, _ in candidates}) == 510
    assert candidates[0][2] == 0


def test_lifepo4_defect_sites_map_to_sorted_poscar_rows(tmp_path):
    symbols = ["Li"] * 16 + ["Fe"] * 16 + ["P"] * 16 + ["O"] * 64
    positions = np.array(
        [[3.0 * x, 3.0 * y, 3.0 * z] for x in range(5) for y in range(5) for z in range(5)][:112]
    )
    parent = Atoms(symbols, positions=positions, cell=[15, 15, 15], pbc=True)
    structure = parent.copy()
    changed_symbols = structure.get_chemical_symbols()
    changed_symbols[5], changed_symbols[22] = "Fe", "Li"
    structure.set_chemical_symbols(changed_symbols)
    poscar_path = tmp_path / "POSCAR"
    _write_poscar(structure, poscar_path, "defect-site-order-test")

    changed, distances = _lifepo4_defect_metadata(parent, structure, poscar_path)
    written = read(poscar_path, format="vasp")
    assert {item["parent_site"] for item in changed} == {5, 22}
    assert len(distances) == 1
    assert distances[0] > 0
    assert any(item["site"] != item["parent_site"] for item in changed)
    assert all(written[item["site"]].symbol == item["to"] for item in changed)


def test_stage1_pbs_files_are_preflight_only(tmp_path):
    _stage_pbs(tmp_path)
    expected = {
        "pbs-dryrun-bcc.pbs": "HydraGNN",
        "pbs-dryrun-fcc.pbs": "HydraGNN",
        "pbs-dryrun-lifepo4.pbs": "HydraGNN",
    }
    for filename, allocation in expected.items():
        script = (tmp_path / filename).read_text()
        assert f"#PBS -A {allocation}" in script
        assert "#PBS -q capacity" in script
        assert "#PBS -l select=1" in script
        assert "#PBS -l walltime=08:00:00" in script
        assert "exit 0" in script
        commands = [line.strip() for line in script.splitlines()]
        assert not any(
            command.startswith(("vasp_std", "mpiexec ", "qsub ")) for command in commands
        )
