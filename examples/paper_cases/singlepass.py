"""
Unified single-pass (non-AL) runner for all paper test cases.
=============================================================
A "single pass" runs the planner -> executor -> uq_gate -> analyst agent graph ONCE on a
supplied/built seed structure: the MLP relaxes it, ranks polymorphs, and (with
``--dft``) the agent may validate the lowest-energy structure with a single DFT
single-point.  There is NO label -> retrain loop — that is what the ``al_*.yaml``
active-learning configs are for.

Use a single pass to:
  * sanity-check that the MLP behaves on a system before committing to AL,
  * get a fast polymorph ranking / feasibility read (phosphorene, MOFs),
  * generate seed structure files that the AL ``seed_source.kind: paths``
    configs then consume.

Cases (mirror the al_*.yaml configs):
    lifepo4      3D  LiFePO4 olivine (battery cathode)
    hea_bcc      3D  NbTaVHfZrTi refractory BCC HEA
    hea_fcc      3D  CrMnFeCoNi Cantor FCC HEA
    phosphorene  2D  black-P monolayer
    cu_bht       2D  Cu-BHT conductive MOF (needs seeds/cu_bht_monolayer.cif)
    zn_formate   3D  Zn(HCOO)2 MOF feasibility

Usage:
    cd /global/cfs/projectdirs/m5216/mlupopa/matsim-agents
    source scripts/setup/perlmutter/setup_matsim_perlmutter.sh

    # one case, MLP only:
    python examples/paper_cases/singlepass.py --case lifepo4

    # with a single DFT validation single-point (VASP):
    python examples/paper_cases/singlepass.py --case zn_formate --dft

    # generate seeds for every case without running the agent:
    python examples/paper_cases/singlepass.py --all --seeds-only

Environment variables (override defaults):
    MLP_LOGDIR   HydraGNN logdir (config.json + checkpoint)   [required to run]
    MLP_CKPT     checkpoint filename            (default: best_model.pt)
    OUT_DIR      output root                    (default: ./out_singlepass)
"""

from __future__ import annotations

import argparse
import os
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.io import read, write
from ase.spacegroup import crystal

# ---------------------------------------------------------------------------
# matsim-agents programmatic API
# ---------------------------------------------------------------------------
from matsim_agents.graph import build_graph
from matsim_agents.state import MatSimState

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ_ROOT  = Path(os.environ.get("PROJ_ROOT", Path(__file__).resolve().parents[2]))
OUT_DIR    = Path(os.environ.get("OUT_DIR", PROJ_ROOT / "out_singlepass"))
MLP_LOGDIR = Path(os.environ.get("MLP_LOGDIR", PROJ_ROOT / "runs/al-models/iter0_logdir"))
MLP_CKPT   = os.environ.get("MLP_CKPT", "best_model.pt")

SEED_DIR = PROJ_ROOT / "examples/paper_cases/seeds"
SEED_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Structure builders — one per case. Each returns a list of (label, Atoms).
# ===========================================================================
def build_lifepo4() -> list[tuple[str, Atoms]]:
    """LiFePO4 olivine, space group Pnma (#62), 28 atoms (4 f.u.)."""
    atoms = crystal(
        symbols=["Li", "Fe", "P", "O", "O", "O"],
        basis=[
            (0.000, 0.000, 0.000),   # Li 4a
            (0.282, 0.250, 0.974),   # Fe 4c
            (0.095, 0.250, 0.418),   # P  4c
            (0.097, 0.250, 0.743),   # O1 4c
            (0.457, 0.250, 0.206),   # O2 4c
            (0.166, 0.046, 0.285),   # O3 8d
        ],
        spacegroup=62,
        cellpar=[10.33, 6.01, 4.69, 90, 90, 90],
    )
    return [("olivine", atoms)]


def _random_alloy(prototype: Atoms, elements: list[str], seed: int) -> Atoms:
    """Assign `elements` as evenly as possible across sites of `prototype`."""
    n = len(prototype)
    base, rem = divmod(n, len(elements))
    counts = [base + (1 if i < rem else 0) for i in range(len(elements))]
    symbols: list[str] = []
    for el, c in zip(elements, counts):
        symbols += [el] * c
    rng = np.random.default_rng(seed)
    rng.shuffle(symbols)
    out = prototype.copy()
    out.set_chemical_symbols(symbols)
    return out


def build_hea_bcc() -> list[tuple[str, Atoms]]:
    """NbTaVHfZrTi equimolar BCC, 24-atom supercell (4 of each element)."""
    proto = make_supercell(bulk("W", "bcc", a=3.30, cubic=True),
                           np.diag([2, 2, 3]))  # 12 cells x 2 = 24 sites
    elements = ["Nb", "Ta", "V", "Hf", "Zr", "Ti"]
    return [("bcc_random", _random_alloy(proto, elements, seed=7))]


def build_hea_fcc() -> list[tuple[str, Atoms]]:
    """CrMnFeCoNi Cantor alloy, FCC, 32-atom supercell (near-equimolar)."""
    proto = make_supercell(bulk("Ni", "fcc", a=3.59, cubic=True),
                           np.diag([2, 2, 2]))  # 8 cells x 4 = 32 sites
    elements = ["Cr", "Mn", "Fe", "Co", "Ni"]
    return [("fcc_random", _random_alloy(proto, elements, seed=13))]


def build_phosphorene(vacuum_A: float = 15.0) -> list[tuple[str, Atoms]]:
    """Phosphorene polymorphs to rank: black-P (Pmna, 4 atoms) and blue-P
    (P-3m1, 2 atoms).

    NOTE: approximate hand-built geometry. For the paper, replace with the
    reference QE structure cited in the collaborator email
    (materialssquare.com/work/43421).
    """
    a, b = 3.30, 4.62
    z_inner = 1.24 / 2.0
    c_total = 1.24 + 4.0 + vacuum_A
    black = Atoms(
        symbols="P4",
        scaled_positions=[
            [0.0, 0.25, 0.5 - z_inner / c_total],
            [0.0, 0.75, 0.5 - z_inner / c_total],
            [0.5, 0.25, 0.5 + z_inner / c_total],
            [0.5, 0.75, 0.5 + z_inner / c_total],
        ],
        cell=[a, b, c_total],
        pbc=[True, True, False],
    )
    # blue-P: trigonal buckled honeycomb (P-3m1, 2 atoms)
    a2, buckling = 3.28, 1.24
    c2 = buckling + 4.0 + vacuum_A
    blue = Atoms(
        symbols="P2",
        scaled_positions=[
            [1 / 3, 2 / 3, 0.5 - buckling / (2.0 * c2)],
            [2 / 3, 1 / 3, 0.5 + buckling / (2.0 * c2)],
        ],
        cell=[[a2, 0, 0],
              [-a2 / 2.0, a2 * (3.0 ** 0.5) / 2.0, 0],
              [0, 0, c2]],
        pbc=[True, True, False],
    )
    return [("black_P", black), ("blue_P", blue)]


def build_cu_bht() -> list[tuple[str, Atoms]]:
    """Cu-BHT (Cu3C6S6) 2D MOF — loaded from a supplied CIF.

    The seed enumerator is inorganic-only, so this MOF must be provided as a
    structure file. Place a validated Cu3C6S6 CIF (e.g. MP mp-630956) at
    examples/paper_cases/seeds/cu_bht_monolayer.cif.
    """
    cif = SEED_DIR / "cu_bht_monolayer.cif"
    if not cif.is_file():
        raise FileNotFoundError(
            f"Cu-BHT seed not found: {cif}\n"
            "Place a validated Cu3C6S6 CIF there (MP id mp-630956 recommended)."
        )
    return [("cu_bht", read(str(cif)))]


def build_zn_formate() -> list[tuple[str, Atoms]]:
    """Zn(HCOO)2 MOF feasibility seeds — alpha (Pna2_1, #33) and beta
    (P2_1 2_1 2_1, #19) polymorphs to rank.

    NOTE: simplified placeholder geometry WITHOUT the formate H atoms. For a
    DFT-quality run, replace with a real CIF (which includes H) before using
    the ``--dft`` path or the al_zn_formate.yaml AL config.
    """
    a, b, c = 9.56, 8.81, 9.56
    scaled = [
        [0.000, 0.000, 0.000], [0.500, 0.500, 0.000],
        [0.000, 0.500, 0.500], [0.500, 0.000, 0.500],            # Zn x4
        [0.120, 0.200, 0.120], [0.380, 0.300, 0.120],
        [0.620, 0.700, 0.880], [0.880, 0.800, 0.880],
        [0.120, 0.700, 0.620], [0.380, 0.800, 0.620],
        [0.620, 0.200, 0.380], [0.880, 0.300, 0.380],            # C x8
        [0.060, 0.250, 0.060], [0.180, 0.150, 0.180],
        [0.440, 0.350, 0.060], [0.320, 0.250, 0.180],
        [0.560, 0.650, 0.940], [0.680, 0.750, 0.820],
        [0.820, 0.850, 0.940], [0.940, 0.750, 0.820],            # O x8
    ]
    symbols = ["Zn"] * 4 + ["C"] * 8 + ["O"] * 8
    alpha = Atoms(symbols=symbols, scaled_positions=scaled, cell=[a, b, c], pbc=True)
    # beta polymorph (P212121, #19): slightly denser packing
    ab, bb, cb = 9.46, 8.75, 9.60
    scaled_beta = [
        [0.002, 0.003, 0.001], [0.498, 0.503, 0.001],
        [0.002, 0.497, 0.499], [0.498, 0.497, 0.499],
        [0.125, 0.195, 0.118], [0.375, 0.305, 0.118],
        [0.625, 0.695, 0.882], [0.875, 0.805, 0.882],
        [0.118, 0.695, 0.625], [0.382, 0.805, 0.625],
        [0.618, 0.195, 0.375], [0.882, 0.305, 0.375],
        [0.062, 0.245, 0.062], [0.188, 0.145, 0.174],
        [0.438, 0.355, 0.062], [0.312, 0.245, 0.174],
        [0.562, 0.645, 0.938], [0.688, 0.755, 0.826],
        [0.812, 0.855, 0.938], [0.938, 0.755, 0.826],
    ]
    beta = Atoms(symbols=symbols, scaled_positions=scaled_beta, cell=[ab, bb, cb], pbc=True)
    return [("alpha", alpha), ("beta", beta)]


# ===========================================================================
# Case registry
# ===========================================================================
CASES = {
    "lifepo4":     dict(builder=build_lifepo4,     is_2d=False),
    "hea_bcc":     dict(builder=build_hea_bcc,     is_2d=False),
    "hea_fcc":     dict(builder=build_hea_fcc,     is_2d=False),
    "phosphorene": dict(builder=build_phosphorene, is_2d=True),
    "cu_bht":      dict(builder=build_cu_bht,      is_2d=True),
    "zn_formate":  dict(builder=build_zn_formate,  is_2d=False),
}


def write_seeds(case: str) -> list[Path]:
    """Build the case structures and write them to the shared seeds/ dir."""
    spec = CASES[case]
    paths: list[Path] = []
    for label, atoms in spec["builder"]():
        p = SEED_DIR / f"{case}_{label}.extxyz"
        write(str(p), atoms)
        comp = dict(Counter(atoms.get_chemical_symbols()))
        print(f"  [{case}] wrote {p.name}  ({len(atoms)} atoms, {comp})")
        paths.append(p)
    return paths


def make_objective(case: str, seed_paths: list[Path], run_dft: bool) -> str:
    spec = CASES[case]
    files = "\n".join(f"  - {p}" for p in seed_paths)
    two_d = (
        " The structure is a 2D monolayer with a vacuum gap along the "
        "non-periodic axis; treat it as a slab (single k-point along the "
        "vacuum direction)." if spec["is_2d"] else ""
    )
    dft = (
        " Then run a single DFT single-point on the lowest-energy relaxed "
        "structure to validate the MLP energy and forces." if run_dft else
        " Report MLP-only results; do not run DFT."
    )
    return (
        f"Relax the following seed structure(s) for case '{case}' with the MLP "
        f"surrogate and rank them by energy per atom:\n{files}\n"
        f"Report the most stable structure, its energy per atom, and the RMSD "
        f"per atom between the initial and relaxed geometry.{two_d}{dft}"
    )


def run_case(case: str, run_dft: bool) -> None:
    print(f"\n=== single pass: {case} ===")
    seed_paths = write_seeds(case)

    graph = build_graph()
    objective = make_objective(case, seed_paths, run_dft)
    state = MatSimState(objective=objective)
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "logdir": str(MLP_LOGDIR),
            "mlp_checkpoint": MLP_CKPT,
            "mlp_device": "cuda",
        }
    }
    final = graph.invoke(state, config=config)
    print(final.get("analysis"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pass (non-AL) runner.")
    parser.add_argument("--case", choices=sorted(CASES), help="Which case to run.")
    parser.add_argument("--all", action="store_true", help="Run every case.")
    parser.add_argument("--dft", action="store_true",
                        help="Also run a single DFT validation single-point.")
    parser.add_argument("--seeds-only", action="store_true",
                        help="Only write seed structure files; do not run the agent.")
    args = parser.parse_args()

    if not args.case and not args.all:
        parser.error("specify --case <name> or --all")

    cases = sorted(CASES) if args.all else [args.case]

    if args.seeds_only:
        for c in cases:
            try:
                write_seeds(c)
            except FileNotFoundError as exc:
                print(f"  [skip] {exc}")
        return

    for c in cases:
        try:
            run_case(c, run_dft=args.dft)
        except FileNotFoundError as exc:
            print(f"  [skip] {c}: {exc}")


if __name__ == "__main__":
    main()
