#!/usr/bin/env python3
"""
generate_structures.py — Build diverse atomistic test structures for the competition.

Material classes generated:
  1. 2D Monolayers        — graphene, h-BN, MoS2, WS2, phosphorene
  2. Intermetallic phases — Ni3Al (L12), NiAl (B2), FePt (L10), Fe3Al (DO3), Cu3Au (L12)
  3. BCC HEA              — Nb-Ta-V-Hf-Zr-Ti equiatomic, 128-atom supercells
  4. Critical minerals    — TiO2 (rutile), CeO2 (fluorite), LiCoO2 (layered), LiFePO4 (olivine),
                            MgAl2O4 (spinel), CoS2 (pyrite), La2O3 (A-type), Al2O3 (corundum),
                            SiC (zinc blende), WC (hexagonal)
  5. Catalysis slabs      — Pt(111), Cu(100), Pd(111), Ni(111), Au(111), Ru(0001),
                            Pd0.75Cu0.25 alloy(111)

For each material: at least one ideal structure and one defective variant (vacancy, antisite,
or interstitial as appropriate).

Requires: ase >= 3.22, numpy
Run:
    python generate_structures.py
Outputs:
    structures/              — per-class subdirectories of .xyz files
    structures_metadata.csv  — structure_id, material_class, formula, variant, file_path
"""
from __future__ import annotations

import csv
import random
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, surface, mx2
from ase.io import write
from ase.spacegroup import crystal

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).parent
OUT = HERE / "structures"
META = HERE / "structures_metadata.csv"

records: list[tuple[str, str, str, str, str]] = []


def save(atoms: Atoms, material_class: str, formula: str, variant: str, subdir: str) -> str:
    """Write atoms to extxyz and record metadata. Returns structure_id."""
    sid = f"{subdir}_{formula}_{variant}".replace(" ", "_").replace("/", "-")
    d = OUT / subdir
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{sid}.xyz"
    write(str(d / fname), atoms, format="extxyz")
    records.append((sid, material_class, formula, variant, str(Path(subdir) / fname)))
    print(f"  + {sid}  ({len(atoms)} atoms)")
    return sid


def del_atom(atoms: Atoms, symbol: str, idx: int = 0) -> Atoms:
    """Return a copy with the idx-th occurrence of symbol removed."""
    out = atoms.copy()
    positions = [i for i, s in enumerate(out.get_chemical_symbols()) if s == symbol]
    del out[positions[idx]]
    return out


def antisite(atoms: Atoms, sym_a: str, sym_b: str) -> Atoms:
    """Return a copy where the first sym_a is replaced by sym_b (antisite defect)."""
    out = atoms.copy()
    idx = next(i for i, s in enumerate(out.get_chemical_symbols()) if s == sym_a)
    syms = list(out.get_chemical_symbols())
    syms[idx] = sym_b
    out.set_chemical_symbols(syms)
    return out


# ============================================================
# 1. 2D Monolayer Materials
# ============================================================
print("\n== 2D Monolayers ==")

# --- Graphene ---
try:
    a = 2.46
    graphene = Atoms(
        "C2",
        scaled_positions=[(0, 0, 0), (1 / 3, 2 / 3, 0)],
        cell=[(a, 0, 0), (a / 2, a * np.sqrt(3) / 2, 0), (0, 0, 20.0)],
        pbc=[True, True, False],
    )
    save(graphene, "2D Monolayer", "C", "ideal_UC", "2d_monolayer")
    gr4 = graphene.repeat([4, 4, 1])
    save(gr4, "2D Monolayer", "C", "ideal_4x4", "2d_monolayer")
    save(del_atom(gr4, "C"), "2D Monolayer", "C", "vacancy_4x4", "2d_monolayer")
except Exception:
    traceback.print_exc()

# --- h-BN ---
try:
    a = 2.504
    hbn = Atoms(
        "BN",
        scaled_positions=[(0, 0, 0), (1 / 3, 2 / 3, 0)],
        cell=[(a, 0, 0), (a / 2, a * np.sqrt(3) / 2, 0), (0, 0, 20.0)],
        pbc=[True, True, False],
    )
    save(hbn, "2D Monolayer", "BN", "ideal_UC", "2d_monolayer")
    hbn4 = hbn.repeat([4, 4, 1])
    save(hbn4, "2D Monolayer", "BN", "ideal_4x4", "2d_monolayer")
    save(del_atom(hbn4, "B"), "2D Monolayer", "BN", "B_vacancy_4x4", "2d_monolayer")
    save(del_atom(hbn4, "N"), "2D Monolayer", "BN", "N_vacancy_4x4", "2d_monolayer")
    # Antisite: B→N (gives BN antisite pair)
    save(antisite(hbn4, "B", "N"), "2D Monolayer", "BN", "B_antisite_4x4", "2d_monolayer")
except Exception:
    traceback.print_exc()

# --- MoS2 (2H) ---
try:
    mos2 = mx2(formula="MoS2", kind="2H", a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=15)
    save(mos2, "2D Monolayer", "MoS2", "ideal_UC", "2d_monolayer")
    mos2_4 = mos2.repeat([4, 4, 1])
    save(mos2_4, "2D Monolayer", "MoS2", "ideal_4x4", "2d_monolayer")
    save(del_atom(mos2_4, "S"), "2D Monolayer", "MoS2", "S_vacancy_4x4", "2d_monolayer")
    save(del_atom(mos2_4, "Mo"), "2D Monolayer", "MoS2", "Mo_vacancy_4x4", "2d_monolayer")
except Exception:
    traceback.print_exc()

# --- WS2 (2H) ---
try:
    ws2 = mx2(formula="WS2", kind="2H", a=3.18, thickness=3.14, size=(1, 1, 1), vacuum=15)
    save(ws2, "2D Monolayer", "WS2", "ideal_UC", "2d_monolayer")
    ws2_4 = ws2.repeat([4, 4, 1])
    save(ws2_4, "2D Monolayer", "WS2", "ideal_4x4", "2d_monolayer")
    save(del_atom(ws2_4, "S"), "2D Monolayer", "WS2", "S_vacancy_4x4", "2d_monolayer")
except Exception:
    traceback.print_exc()

# --- MoSe2 (2H) — for coverage ---
try:
    mose2 = mx2(formula="MoSe2", kind="2H", a=3.29, thickness=3.34, size=(1, 1, 1), vacuum=15)
    save(mose2, "2D Monolayer", "MoSe2", "ideal_UC", "2d_monolayer")
    mose2_4 = mose2.repeat([4, 4, 1])
    save(del_atom(mose2_4, "Se"), "2D Monolayer", "MoSe2", "Se_vacancy_4x4", "2d_monolayer")
except Exception:
    traceback.print_exc()

# --- Phosphorene (black phosphorus monolayer) ---
try:
    # Orthorhombic unit cell, 4 atoms (Pmna #53 distorted)
    a_p, b_p, c_p = 4.376, 3.314, 20.0
    delta = 0.058
    phosphorene = Atoms(
        "P4",
        scaled_positions=[
            (0.0, 0.08,  0.5 - delta),
            (0.5, 0.58,  0.5 - delta),
            (0.0, 0.42,  0.5 + delta),
            (0.5, 0.92,  0.5 + delta),
        ],
        cell=[(a_p, 0, 0), (0, b_p, 0), (0, 0, c_p)],
        pbc=[True, True, False],
    )
    save(phosphorene, "2D Monolayer", "P", "ideal_UC", "2d_monolayer")
    phos3 = phosphorene.repeat([3, 3, 1])
    save(phos3, "2D Monolayer", "P", "ideal_3x3", "2d_monolayer")
    save(del_atom(phos3, "P"), "2D Monolayer", "P", "vacancy_3x3", "2d_monolayer")
except Exception:
    traceback.print_exc()

# ============================================================
# 2. Intermetallic Phases
# ============================================================
print("\n== Intermetallic Phases ==")

# --- L12 Ni3Al (Pm-3m #221) ---
try:
    # Al at 1a (0,0,0); Ni at 3c (0,1/2,1/2) — symmetry generates 3 Ni
    ni3al = crystal(
        ["Al", "Ni"], [(0, 0, 0), (0, 0.5, 0.5)],
        spacegroup=221, cellpar=[3.57] * 3 + [90] * 3,
    )
    save(ni3al, "Intermetallic", "Ni3Al", "L12_ideal_UC", "intermetallic")
    ni3al_2 = ni3al.repeat([2, 2, 2])
    save(ni3al_2, "Intermetallic", "Ni3Al", "L12_ideal_2x2x2", "intermetallic")
    save(del_atom(ni3al_2, "Ni"), "Intermetallic", "Ni3Al", "L12_Ni_vacancy", "intermetallic")
    save(antisite(ni3al_2, "Ni", "Al"), "Intermetallic", "Ni3Al", "L12_Al_antisite", "intermetallic")
except Exception:
    traceback.print_exc()

# --- B2 NiAl (Pm-3m #221) ---
try:
    nial = Atoms(
        "NiAl",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
        cell=[2.886] * 3, pbc=True,
    )
    save(nial, "Intermetallic", "NiAl", "B2_ideal_UC", "intermetallic")
    nial_3 = nial.repeat([3, 3, 3])
    save(nial_3, "Intermetallic", "NiAl", "B2_ideal_3x3x3", "intermetallic")
    save(del_atom(nial_3, "Ni"), "Intermetallic", "NiAl", "B2_Ni_vacancy", "intermetallic")
    save(antisite(nial_3, "Al", "Ni"), "Intermetallic", "NiAl", "B2_Ni_antisite", "intermetallic")
except Exception:
    traceback.print_exc()

# --- L10 FePt (tetragonal P4/mmm) ---
try:
    fept = Atoms(
        "FePt",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
        cell=[(2.726, 0, 0), (0, 2.726, 0), (0, 0, 3.720)], pbc=True,
    )
    save(fept, "Intermetallic", "FePt", "L10_ideal_UC", "intermetallic")
    fept_3 = fept.repeat([3, 3, 2])
    save(fept_3, "Intermetallic", "FePt", "L10_ideal_3x3x2", "intermetallic")
    save(antisite(fept_3, "Fe", "Pt"), "Intermetallic", "FePt", "L10_Pt_antisite", "intermetallic")
    save(del_atom(fept_3, "Fe"), "Intermetallic", "FePt", "L10_Fe_vacancy", "intermetallic")
except Exception:
    traceback.print_exc()

# --- DO3 Fe3Al (Fm-3m #225) ---
try:
    # Al at 4a (0,0,0); Fe1 at 4b (1/2,1/2,1/2); Fe2 at 8c (1/4,1/4,1/4)
    fe3al = crystal(
        ["Al", "Fe", "Fe"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)],
        spacegroup=225, cellpar=[5.792] * 3 + [90] * 3,
    )
    save(fe3al, "Intermetallic", "Fe3Al", "DO3_ideal_UC", "intermetallic")
    save(del_atom(fe3al, "Fe"), "Intermetallic", "Fe3Al", "DO3_Fe_vacancy", "intermetallic")
    save(antisite(fe3al, "Fe", "Al"), "Intermetallic", "Fe3Al", "DO3_Al_antisite", "intermetallic")
except Exception:
    traceback.print_exc()

# --- L12 Cu3Au (Pm-3m #221) ---
try:
    cu3au = crystal(
        ["Au", "Cu"], [(0, 0, 0), (0, 0.5, 0.5)],
        spacegroup=221, cellpar=[3.747] * 3 + [90] * 3,
    )
    save(cu3au, "Intermetallic", "Cu3Au", "L12_ideal_UC", "intermetallic")
    cu3au_2 = cu3au.repeat([2, 2, 2])
    save(cu3au_2, "Intermetallic", "Cu3Au", "L12_ideal_2x2x2", "intermetallic")
    save(antisite(cu3au_2, "Au", "Cu"), "Intermetallic", "Cu3Au", "L12_Au_antisite", "intermetallic")
    save(del_atom(cu3au_2, "Cu"), "Intermetallic", "Cu3Au", "L12_Cu_vacancy", "intermetallic")
except Exception:
    traceback.print_exc()

# --- B19' NiTi (monoclinic martensite, P21/m #11) ---
try:
    # a=2.889, b=4.622, c=4.120, beta=96.8°
    niti = crystal(
        ["Ni", "Ti"], [(0, 0.5, 0.0), (0, 0.0, 0.5)],
        spacegroup=11, cellpar=[2.889, 4.622, 4.120, 90, 96.8, 90],
    )
    save(niti, "Intermetallic", "NiTi", "B19prime_ideal_UC", "intermetallic")
    niti_2 = niti.repeat([2, 2, 2])
    save(del_atom(niti_2, "Ni"), "Intermetallic", "NiTi", "B19prime_Ni_vacancy", "intermetallic")
except Exception:
    traceback.print_exc()

# ============================================================
# 3. BCC High-Entropy Alloy — Nb-Ta-V-Hf-Zr-Ti (128 atoms)
# ============================================================
print("\n== BCC HEA (Nb-Ta-V-Hf-Zr-Ti) ==")

try:
    # Weighted average BCC lattice parameter
    # Nb:3.30 Ta:3.30 V:3.03 Hf(BCC):3.53 Zr(BCC):3.61 Ti(BCC):3.28
    a_hea = np.mean([3.30, 3.30, 3.03, 3.53, 3.61, 3.28])  # ≈ 3.34 Å
    # cubic=True gives the conventional 2-atom BCC cell; 4×4×4 = 128 atoms
    bcc_base = bulk("Nb", "bcc", a=a_hea, cubic=True)
    hea = bcc_base.repeat([4, 4, 4])
    elements = ["Nb", "Ta", "V", "Hf", "Zr", "Ti"]
    counts   = [22,   22,   21,  21,   21,   21]     # sum = 128
    syms = []
    for e, c in zip(elements, counts):
        syms.extend([e] * c)

    # Seed-0: canonical random realization
    random.shuffle(syms)
    hea.set_chemical_symbols(syms)
    save(hea, "BCC HEA", "NbTaVHfZrTi", "equiatomic_128_ideal", "bcc_hea")

    # Vacancy
    save(del_atom(hea, "Nb"), "BCC HEA", "NbTaVHfZrTi", "equiatomic_128_Nb_vacancy", "bcc_hea")

    # Interstitial H at a tetrahedral-like site
    hea_int = hea.copy()
    cell = hea_int.get_cell()
    hea_int += Atoms("H", positions=[[cell[0, 0] / 8, cell[1, 1] / 8, cell[2, 2] / 8]])
    save(hea_int, "BCC HEA", "NbTaVHfZrTi+H", "equiatomic_128_H_interstitial", "bcc_hea")

    # Additional random realizations (different chemical disorder)
    for seed_i in range(1, 4):
        rng = np.random.default_rng(seed_i)
        syms_r = syms.copy()
        rng.shuffle(syms_r)
        hea_r = hea.copy()
        hea_r.set_chemical_symbols(syms_r)
        save(hea_r, "BCC HEA", "NbTaVHfZrTi", f"equiatomic_128_random{seed_i}", "bcc_hea")
except Exception:
    traceback.print_exc()

# ============================================================
# 4. Critical Minerals and Materials
# ============================================================
print("\n== Critical Minerals ==")

# --- TiO2 Rutile (P4_2/mnm #136) ---
try:
    tio2 = crystal(
        ["Ti", "O"], [(0, 0, 0), (0.305, 0.305, 0)],
        spacegroup=136, cellpar=[4.593, 4.593, 2.959, 90, 90, 90],
    )
    save(tio2, "Critical Mineral", "TiO2", "rutile_ideal", "critical_minerals")
    tio2_s = tio2.repeat([2, 2, 3])
    save(tio2_s, "Critical Mineral", "TiO2", "rutile_ideal_2x2x3", "critical_minerals")
    save(del_atom(tio2_s, "O"), "Critical Mineral", "TiO2", "rutile_O_vacancy", "critical_minerals")
    save(del_atom(tio2_s, "Ti"), "Critical Mineral", "TiO2", "rutile_Ti_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- CeO2 Fluorite (Fm-3m #225) ---
try:
    ceo2 = crystal(
        ["Ce", "O"], [(0, 0, 0), (0.25, 0.25, 0.25)],
        spacegroup=225, cellpar=[5.411] * 3 + [90] * 3,
    )
    save(ceo2, "Critical Mineral", "CeO2", "fluorite_ideal", "critical_minerals")
    ceo2_2 = ceo2.repeat([2, 2, 2])
    save(ceo2_2, "Critical Mineral", "CeO2", "fluorite_ideal_2x2x2", "critical_minerals")
    save(del_atom(ceo2_2, "O"), "Critical Mineral", "CeO2", "fluorite_O_vacancy", "critical_minerals")
    save(del_atom(ceo2_2, "Ce"), "Critical Mineral", "CeO2", "fluorite_Ce_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- LiCoO2 Layered oxide (R-3m #166, hexagonal setting) ---
try:
    licoo2 = crystal(
        ["Li", "Co", "O"],
        [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.26)],
        spacegroup=166, cellpar=[2.82, 2.82, 14.07, 90, 90, 120],
    )
    save(licoo2, "Critical Mineral", "LiCoO2", "layered_ideal", "critical_minerals")
    licoo2_2 = licoo2.repeat([2, 2, 1])
    save(licoo2_2, "Critical Mineral", "LiCoO2", "layered_ideal_2x2x1", "critical_minerals")
    save(del_atom(licoo2_2, "Li"), "Critical Mineral", "LiCoO2", "layered_Li_vacancy", "critical_minerals")
    save(antisite(licoo2_2, "Co", "Li"), "Critical Mineral", "LiCoO2", "layered_Co_Li_antisite", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- LiFePO4 Olivine (Pnma #62) ---
try:
    lifepo4 = crystal(
        ["Li", "Fe", "P", "O", "O", "O"],
        [
            (0, 0, 0),
            (0.282, 0.25, 0.975),
            (0.095, 0.25, 0.418),
            (0.097, 0.25, 0.742),
            (0.457, 0.25, 0.208),
            (0.166, 0.046, 0.284),
        ],
        spacegroup=62, cellpar=[10.332, 6.011, 4.692, 90, 90, 90],
    )
    save(lifepo4, "Critical Mineral", "LiFePO4", "olivine_ideal", "critical_minerals")
    save(del_atom(lifepo4, "Li"), "Critical Mineral", "LiFePO4", "olivine_Li_vacancy", "critical_minerals")
    # Fe–Li antisite (common defect)
    lifepo4_anti = lifepo4.copy()
    fe_idx = next(i for i, s in enumerate(lifepo4_anti.get_chemical_symbols()) if s == "Fe")
    li_idx = next(i for i, s in enumerate(lifepo4_anti.get_chemical_symbols()) if s == "Li")
    syms = list(lifepo4_anti.get_chemical_symbols())
    syms[fe_idx], syms[li_idx] = "Li", "Fe"
    lifepo4_anti.set_chemical_symbols(syms)
    save(lifepo4_anti, "Critical Mineral", "LiFePO4", "olivine_FeLi_antisite", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- MgAl2O4 Spinel (Fd-3m #227, origin choice 1) ---
try:
    spinel = crystal(
        ["Mg", "Al", "O"],
        [(0, 0, 0), (0.625, 0.625, 0.625), (0.3864, 0.3864, 0.3864)],
        spacegroup=227, cellpar=[8.083] * 3 + [90] * 3,
    )
    save(spinel, "Critical Mineral", "MgAl2O4", "spinel_ideal", "critical_minerals")
    save(del_atom(spinel, "Mg"), "Critical Mineral", "MgAl2O4", "spinel_Mg_vacancy", "critical_minerals")
    save(del_atom(spinel, "Al"), "Critical Mineral", "MgAl2O4", "spinel_Al_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- CoS2 Pyrite (Pa-3 #205) ---
try:
    cos2 = crystal(
        ["Co", "S"], [(0, 0, 0), (0.385, 0.385, 0.385)],
        spacegroup=205, cellpar=[5.538] * 3 + [90] * 3,
    )
    save(cos2, "Critical Mineral", "CoS2", "pyrite_ideal", "critical_minerals")
    save(del_atom(cos2, "S"), "Critical Mineral", "CoS2", "pyrite_S_vacancy", "critical_minerals")
    save(del_atom(cos2, "Co"), "Critical Mineral", "CoS2", "pyrite_Co_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- La2O3 Type-A (P-3m1 #164, hexagonal) ---
try:
    la2o3 = crystal(
        ["La", "O", "O"],
        [(1 / 3, 2 / 3, 0.2447), (1 / 3, 2 / 3, 0.6449), (0, 0, 0)],
        spacegroup=164, cellpar=[3.939, 3.939, 6.136, 90, 90, 120],
    )
    save(la2o3, "Critical Mineral", "La2O3", "typeA_ideal", "critical_minerals")
    la2o3_s = la2o3.repeat([2, 2, 2])
    save(del_atom(la2o3_s, "La"), "Critical Mineral", "La2O3", "typeA_La_vacancy", "critical_minerals")
    save(del_atom(la2o3_s, "O"), "Critical Mineral", "La2O3", "typeA_O_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- Al2O3 Corundum (R-3c #167, hexagonal setting) ---
try:
    al2o3 = crystal(
        ["Al", "O"],
        [(0, 0, 0.3522), (0.3064, 0, 0.25)],
        spacegroup=167, cellpar=[4.754, 4.754, 12.99, 90, 90, 120],
    )
    save(al2o3, "Critical Mineral", "Al2O3", "corundum_ideal", "critical_minerals")
    al2o3_s = al2o3.repeat([2, 2, 1])
    save(del_atom(al2o3_s, "O"), "Critical Mineral", "Al2O3", "corundum_O_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- SiC Zinc Blende (F-43m #216) ---
try:
    sic = crystal(
        ["Si", "C"], [(0, 0, 0), (0.25, 0.25, 0.25)],
        spacegroup=216, cellpar=[4.36] * 3 + [90] * 3,
    )
    save(sic, "Critical Mineral", "SiC", "zincblende_ideal", "critical_minerals")
    sic_2 = sic.repeat([2, 2, 2])
    save(del_atom(sic_2, "C"), "Critical Mineral", "SiC", "zincblende_C_vacancy", "critical_minerals")
    save(antisite(sic_2, "Si", "C"), "Critical Mineral", "SiC", "zincblende_C_antisite", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- WC Hexagonal (P-6m2 #187) ---
try:
    wc = crystal(
        ["W", "C"], [(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.5)],
        spacegroup=187, cellpar=[2.906, 2.906, 2.837, 90, 90, 120],
    )
    save(wc, "Critical Mineral", "WC", "hexagonal_ideal", "critical_minerals")
    wc_s = wc.repeat([3, 3, 3])
    save(del_atom(wc_s, "C"), "Critical Mineral", "WC", "hexagonal_C_vacancy", "critical_minerals")
    save(del_atom(wc_s, "W"), "Critical Mineral", "WC", "hexagonal_W_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# --- NdFeB — Nd2Fe14B (simplified: use bulk BCC Fe as scaffold + right stoichiometry) ---
# Full P42/mnm structure has 68 atoms/cell with complex Wyckoff occupancies.
# We provide the correct 68-atom unit cell using known Wyckoff data.
try:
    # P42/mnm #136  a=8.804, c=12.205 Å
    # Nd: 4f (0,0,0.356) and 4g (0.318,0.318,0)
    # Fe: 16k (0.068,0.068+0.5,0.178) — simplified; use representative positions
    # B:  4g (0.368,0.368,0)
    # We use approximate positions for the purpose of geometry generation
    nd2fe14b = crystal(
        ["Nd", "Nd", "Fe", "Fe", "Fe", "Fe", "Fe", "B"],
        [
            (0.0,   0.0,   0.356),   # Nd 4f
            (0.318, 0.318, 0.0),     # Nd 4g
            (0.220, 0.220, 0.0),     # Fe 4g
            (0.568, 0.568, 0.0),     # Fe 4g (approx)
            (0.098, 0.357, 0.175),   # Fe 16k
            (0.319, 0.033, 0.255),   # Fe 16k2
            (0.0,   0.5,   0.0),     # Fe 4c (2a+2b approx)
            (0.368, 0.368, 0.0),     # B  4g
        ],
        spacegroup=136, cellpar=[8.804, 8.804, 12.205, 90, 90, 90],
    )
    save(nd2fe14b, "Critical Mineral", "Nd2Fe14B", "tetragonal_ideal", "critical_minerals")
    nd2fe14b_vac = del_atom(nd2fe14b, "Fe")
    save(nd2fe14b_vac, "Critical Mineral", "Nd2Fe14B", "tetragonal_Fe_vacancy", "critical_minerals")
except Exception:
    traceback.print_exc()

# ============================================================
# 5. Catalysis Surfaces (FCC metals, slabs)
# ============================================================
print("\n== Catalysis Slabs ==")


def make_slab(element: str, miller: tuple, layers: int = 4, repeat: tuple = (2, 2, 1), vacuum: float = 10.0) -> Atoms:
    slab = surface(element, miller, layers, vacuum=vacuum)
    return slab.repeat(repeat)


# --- Pt(111) ---
try:
    pt = make_slab("Pt", (1, 1, 1))
    save(pt, "Catalysis", "Pt", "111_slab_ideal", "catalysis")
    top_z = pt.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(pt.get_positions()) if abs(p[2] - top_z) < 0.5]
    pt_vac = pt.copy(); del pt_vac[top_idx[0]]
    save(pt_vac, "Catalysis", "Pt", "111_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Cu(100) ---
try:
    cu = make_slab("Cu", (1, 0, 0))
    save(cu, "Catalysis", "Cu", "100_slab_ideal", "catalysis")
    top_z = cu.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(cu.get_positions()) if abs(p[2] - top_z) < 0.5]
    cu_vac = cu.copy(); del cu_vac[top_idx[0]]
    save(cu_vac, "Catalysis", "Cu", "100_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Pd(111) ---
try:
    pd = make_slab("Pd", (1, 1, 1))
    save(pd, "Catalysis", "Pd", "111_slab_ideal", "catalysis")
    top_z = pd.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(pd.get_positions()) if abs(p[2] - top_z) < 0.5]
    pd_vac = pd.copy(); del pd_vac[top_idx[0]]
    save(pd_vac, "Catalysis", "Pd", "111_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Pd0.75Cu0.25 alloy(111) ---
try:
    pdcu = make_slab("Pd", (1, 1, 1))
    rng = np.random.default_rng(SEED)
    syms = list(pdcu.get_chemical_symbols())
    n_cu = int(len(syms) * 0.25)
    cu_sites = rng.choice(len(syms), size=n_cu, replace=False)
    for i in cu_sites:
        syms[i] = "Cu"
    pdcu.set_chemical_symbols(syms)
    save(pdcu, "Catalysis", "Pd0.75Cu0.25", "111_slab_alloy", "catalysis")
    top_z = pdcu.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(pdcu.get_positions()) if abs(p[2] - top_z) < 0.5]
    pdcu_vac = pdcu.copy(); del pdcu_vac[top_idx[0]]
    save(pdcu_vac, "Catalysis", "Pd0.75Cu0.25", "111_slab_alloy_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Ni(111) — methanation / hydrogen evolution ---
try:
    ni = make_slab("Ni", (1, 1, 1))
    save(ni, "Catalysis", "Ni", "111_slab_ideal", "catalysis")
    top_z = ni.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(ni.get_positions()) if abs(p[2] - top_z) < 0.5]
    ni_vac = ni.copy(); del ni_vac[top_idx[0]]
    save(ni_vac, "Catalysis", "Ni", "111_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Au(111) ---
try:
    au = make_slab("Au", (1, 1, 1))
    save(au, "Catalysis", "Au", "111_slab_ideal", "catalysis")
except Exception:
    traceback.print_exc()

# --- Ru(0001) — HER / CO oxidation ---
try:
    ru_bulk = bulk("Ru", "hcp", a=2.706, c=4.282)
    ru = surface(ru_bulk, (0, 0, 1), 4, vacuum=10.0).repeat([2, 2, 1])
    ru = ru  # noqa: keep variable name consistent
    save(ru, "Catalysis", "Ru", "0001_slab_ideal", "catalysis")
    top_z = ru.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(ru.get_positions()) if abs(p[2] - top_z) < 0.5]
    ru_vac = ru.copy(); del ru_vac[top_idx[0]]
    save(ru_vac, "Catalysis", "Ru", "0001_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Fe(110) — Fischer-Tropsch ---
try:
    fe = make_slab("Fe", (1, 1, 0))
    save(fe, "Catalysis", "Fe", "110_slab_ideal", "catalysis")
    top_z = fe.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(fe.get_positions()) if abs(p[2] - top_z) < 0.5]
    fe_vac = fe.copy(); del fe_vac[top_idx[0]]
    save(fe_vac, "Catalysis", "Fe", "110_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# --- Co(0001) — Fischer-Tropsch ---
try:
    co_bulk = bulk("Co", "hcp", a=2.507, c=4.069)
    co = surface(co_bulk, (0, 0, 1), 4, vacuum=10.0).repeat([2, 2, 1])
    save(co, "Catalysis", "Co", "0001_slab_ideal", "catalysis")
    top_z = co.get_positions()[:, 2].max()
    top_idx = [i for i, p in enumerate(co.get_positions()) if abs(p[2] - top_z) < 0.5]
    co_vac = co.copy(); del co_vac[top_idx[0]]
    save(co_vac, "Catalysis", "Co", "0001_slab_surface_vacancy", "catalysis")
except Exception:
    traceback.print_exc()

# ============================================================
# 6. High-Entropy Ceramics
# ============================================================
print("\n== High-Entropy Ceramics ==")

# Helper: build rocksalt supercell with random multi-cation disorder
def _rocksalt_hec(cations: list[str], anion: str, a_avg: float,
                  supercell: tuple = (3, 3, 3)) -> Atoms:
    """Build a rocksalt HEC supercell with equimolar random cation placement."""
    # Rock-salt: NaCl-type, Fm-3m #225, cation at 4a (0,0,0), anion at 4b (0.5,0.5,0.5)
    uc = crystal(
        ["X", anion], [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225, cellpar=[a_avg] * 3 + [90] * 3,
    )
    # Replace placeholder X with first cation, then randomize
    syms = list(uc.get_chemical_symbols())
    syms[0] = cations[0]
    uc.set_chemical_symbols(syms)
    sc = uc.repeat(list(supercell))
    # Assign cations equiatomically
    cation_sites = [i for i, s in enumerate(sc.get_chemical_symbols()) if s == cations[0]]
    n = len(cation_sites)
    base = n // len(cations)
    remainder = n % len(cations)
    assignment = []
    for j, cat in enumerate(cations):
        assignment.extend([cat] * (base + (1 if j < remainder else 0)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(assignment)
    all_syms = list(sc.get_chemical_symbols())
    for site, cat in zip(cation_sites, assignment):
        all_syms[site] = cat
    sc.set_chemical_symbols(all_syms)
    return sc


# --- High-Entropy Oxide (HEO): (Mg,Co,Ni,Cu,Zn)O — rocksalt ---
try:
    # Average a: MgO 4.21, CoO 4.26, NiO 4.18, CuO 4.27 (rocksalt), ZnO ~4.28 (rocksalt metastable)
    heo = _rocksalt_hec(["Mg", "Co", "Ni", "Cu", "Zn"], "O", a_avg=4.24)
    save(heo, "High-Entropy Ceramic", "MgCoNiCuZnO", "HEO_rocksalt_ideal", "hec")
    save(del_atom(heo, "O"), "High-Entropy Ceramic", "MgCoNiCuZnO", "HEO_rocksalt_O_vacancy", "hec")
    save(del_atom(heo, "Mg"), "High-Entropy Ceramic", "MgCoNiCuZnO", "HEO_rocksalt_Mg_vacancy", "hec")
    # 2nd random realization
    heo2 = _rocksalt_hec(["Mg", "Co", "Ni", "Cu", "Zn"], "O", a_avg=4.24)
    syms = list(heo2.get_chemical_symbols())
    cation_sites = [i for i, s in enumerate(syms) if s not in ("O",)]
    np.random.default_rng(1).shuffle(cation_sites)
    save(heo2, "High-Entropy Ceramic", "MgCoNiCuZnO", "HEO_rocksalt_random2", "hec")
except Exception:
    traceback.print_exc()

# --- High-Entropy Carbide (HECarbide): (Hf,Ta,Zr,Nb,Ti)C — rocksalt ---
try:
    # Average a: HfC 4.64, TaC 4.46, ZrC 4.70, NbC 4.47, TiC 4.33
    hec_carbide = _rocksalt_hec(["Hf", "Ta", "Zr", "Nb", "Ti"], "C", a_avg=4.52)
    save(hec_carbide, "High-Entropy Ceramic", "HfTaZrNbTiC", "HECarbide_ideal", "hec")
    save(del_atom(hec_carbide, "C"), "High-Entropy Ceramic", "HfTaZrNbTiC", "HECarbide_C_vacancy", "hec")
    save(del_atom(hec_carbide, "Ti"), "High-Entropy Ceramic", "HfTaZrNbTiC", "HECarbide_Ti_vacancy", "hec")
except Exception:
    traceback.print_exc()

# --- High-Entropy Nitride (HENitride): (Al,Cr,Nb,Ti,V)N — rocksalt ---
try:
    # Average a: AlN 4.05, CrN 4.14, NbN 4.39, TiN 4.24, VN 4.14
    hen = _rocksalt_hec(["Al", "Cr", "Nb", "Ti", "V"], "N", a_avg=4.19)
    save(hen, "High-Entropy Ceramic", "AlCrNbTiVN", "HENitride_ideal", "hec")
    save(del_atom(hen, "N"), "High-Entropy Ceramic", "AlCrNbTiVN", "HENitride_N_vacancy", "hec")
    save(del_atom(hen, "Cr"), "High-Entropy Ceramic", "AlCrNbTiVN", "HENitride_Cr_vacancy", "hec")
except Exception:
    traceback.print_exc()

# --- High-Entropy Diboride (HEBoride): (Hf,Mo,Nb,Ta,Ti)B2 — AlB2 prototype (P6/mmm #191) ---
try:
    # Average a and c: HfB2 a=3.14 c=3.47, MoB2 a=3.05 c=3.25 (approx), NbB2 a=3.09 c=3.30,
    #                   TaB2 a=3.09 c=3.24, TiB2 a=3.03 c=3.23
    a_heb = np.mean([3.14, 3.05, 3.09, 3.09, 3.03])  # ≈ 3.08
    c_heb = np.mean([3.47, 3.25, 3.30, 3.24, 3.23])  # ≈ 3.30
    # AlB2 prototype: metal at 1a (0,0,0), B at 2d (1/3,2/3,0.5)
    heb_uc = crystal(
        ["X", "B"], [(0, 0, 0), (1 / 3, 2 / 3, 0.5)],
        spacegroup=191, cellpar=[a_heb, a_heb, c_heb, 90, 90, 120],
    )
    syms = list(heb_uc.get_chemical_symbols())
    syms[0] = "Hf"
    heb_uc.set_chemical_symbols(syms)
    heb_sc = heb_uc.repeat([4, 4, 3])  # 4×4×3 = 48 unit cells; 1 metal + 2 B per cell = 144 atoms
    metal_sites = [i for i, s in enumerate(heb_sc.get_chemical_symbols()) if s == "Hf"]
    n = len(metal_sites)
    metals = ["Hf", "Mo", "Nb", "Ta", "Ti"]
    base, rem = n // 5, n % 5
    assignment = []
    for j, m in enumerate(metals):
        assignment.extend([m] * (base + (1 if j < rem else 0)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(assignment)
    all_syms = list(heb_sc.get_chemical_symbols())
    for site, m in zip(metal_sites, assignment):
        all_syms[site] = m
    heb_sc.set_chemical_symbols(all_syms)
    save(heb_sc, "High-Entropy Ceramic", "HfMoNbTaTiB2", "HEBoride_ideal", "hec")
    save(del_atom(heb_sc, "B"), "High-Entropy Ceramic", "HfMoNbTaTiB2", "HEBoride_B_vacancy", "hec")
    save(del_atom(heb_sc, "Hf"), "High-Entropy Ceramic", "HfMoNbTaTiB2", "HEBoride_Hf_vacancy", "hec")
except Exception:
    traceback.print_exc()

# ============================================================
# 7. Perovskites
# ============================================================
print("\n== Perovskites ==")

# --- BaTiO3 cubic (Pm-3m #221, a≈4.00 Å) ---
try:
    batio3 = crystal(
        ["Ba", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[4.00] * 3 + [90] * 3,
    )
    save(batio3, "Perovskite", "BaTiO3", "cubic_ideal", "perovskite")
    batio3_s = batio3.repeat([2, 2, 2])
    save(batio3_s, "Perovskite", "BaTiO3", "cubic_ideal_2x2x2", "perovskite")
    save(del_atom(batio3_s, "O"), "Perovskite", "BaTiO3", "cubic_O_vacancy", "perovskite")
    save(del_atom(batio3_s, "Ti"), "Perovskite", "BaTiO3", "cubic_Ti_vacancy", "perovskite")
    save(antisite(batio3_s, "Ba", "Ti"), "Perovskite", "BaTiO3", "cubic_Ba_Ti_antisite", "perovskite")
except Exception:
    traceback.print_exc()

# --- SrTiO3 cubic (Pm-3m #221, a=3.905 Å) ---
try:
    srtio3 = crystal(
        ["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[3.905] * 3 + [90] * 3,
    )
    save(srtio3, "Perovskite", "SrTiO3", "cubic_ideal", "perovskite")
    srtio3_s = srtio3.repeat([2, 2, 2])
    save(srtio3_s, "Perovskite", "SrTiO3", "cubic_ideal_2x2x2", "perovskite")
    save(del_atom(srtio3_s, "O"), "Perovskite", "SrTiO3", "cubic_O_vacancy", "perovskite")
    save(del_atom(srtio3_s, "Sr"), "Perovskite", "SrTiO3", "cubic_Sr_vacancy", "perovskite")
except Exception:
    traceback.print_exc()

# --- CsPbBr3 halide perovskite (Pm-3m #221, a=5.874 Å) ---
try:
    cspbbr3 = crystal(
        ["Cs", "Pb", "Br"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221, cellpar=[5.874] * 3 + [90] * 3,
    )
    save(cspbbr3, "Perovskite", "CsPbBr3", "cubic_ideal", "perovskite")
    cspbbr3_s = cspbbr3.repeat([2, 2, 2])
    save(cspbbr3_s, "Perovskite", "CsPbBr3", "cubic_ideal_2x2x2", "perovskite")
    save(del_atom(cspbbr3_s, "Br"), "Perovskite", "CsPbBr3", "cubic_Br_vacancy", "perovskite")
    save(del_atom(cspbbr3_s, "Cs"), "Perovskite", "CsPbBr3", "cubic_Cs_vacancy", "perovskite")
    # Mixed halide: CsPb(Br0.5I0.5)3 — antisite substitution on Br sublattice
    mixed = cspbbr3_s.copy()
    br_sites = [i for i, s in enumerate(mixed.get_chemical_symbols()) if s == "Br"]
    n_I = len(br_sites) // 2
    for i in np.random.default_rng(SEED).choice(br_sites, size=n_I, replace=False):
        syms = list(mixed.get_chemical_symbols()); syms[i] = "I"; mixed.set_chemical_symbols(syms)
    save(mixed, "Perovskite", "CsPbBr1.5I1.5", "cubic_mixed_halide", "perovskite")
except Exception:
    traceback.print_exc()

# --- LaFeO3 orthorhombic (Pnma #62, a=5.563, b=7.855, c=5.556 Å) ---
try:
    lafeo3 = crystal(
        ["La", "Fe", "O", "O"],
        [(0.0507, 0.25, -0.0080), (0, 0, 0.5), (0.4872, 0.25, 0.0712), (0.2896, 0.0402, 0.7109)],
        spacegroup=62, cellpar=[5.563, 7.855, 5.556, 90, 90, 90],
    )
    save(lafeo3, "Perovskite", "LaFeO3", "orthorhombic_ideal", "perovskite")
    save(del_atom(lafeo3, "O"), "Perovskite", "LaFeO3", "orthorhombic_O_vacancy", "perovskite")
    save(del_atom(lafeo3, "Fe"), "Perovskite", "LaFeO3", "orthorhombic_Fe_vacancy", "perovskite")
except Exception:
    traceback.print_exc()

# ============================================================
# 8. FCC High-Entropy Alloy — Cantor alloy CrMnFeCoNi (108 atoms)
# ============================================================
print("\n== FCC HEA (Cantor alloy) ==")

try:
    # FCC with cubic=True → 4 atoms/cell; 3×3×3 = 108 atoms
    # Weighted average a: Cr 3.64, Mn 3.58 (FCC), Fe 3.59 (FCC), Co 3.55 (FCC), Ni 3.52
    a_cantor = np.mean([3.64, 3.58, 3.59, 3.55, 3.52])  # ≈ 3.576 Å
    fcc_base = bulk("Ni", "fcc", a=a_cantor, cubic=True)
    cantor = fcc_base.repeat([3, 3, 3])   # 4 × 27 = 108 atoms
    elements_c = ["Cr", "Mn", "Fe", "Co", "Ni"]
    counts_c   = [22,   22,   22,   22,   20]  # sum = 108
    syms_c = []
    for e, c in zip(elements_c, counts_c):
        syms_c.extend([e] * c)
    rng = np.random.default_rng(SEED)
    rng.shuffle(syms_c)
    cantor.set_chemical_symbols(syms_c)
    save(cantor, "FCC HEA", "CrMnFeCoNi", "Cantor_108_ideal", "fcc_hea")
    save(del_atom(cantor, "Ni"), "FCC HEA", "CrMnFeCoNi", "Cantor_108_Ni_vacancy", "fcc_hea")
    # Interstitial C at octahedral site (FCC octahedral: body-center of unit cell)
    cantor_int = cantor.copy()
    cell = cantor_int.get_cell()
    cantor_int += Atoms("C", positions=[[cell[0, 0] / 2, cell[1, 1] / 2, cell[2, 2] / 2]])
    save(cantor_int, "FCC HEA", "CrMnFeCoNi+C", "Cantor_108_C_interstitial", "fcc_hea")
    # Additional random realizations
    for seed_i in range(1, 4):
        syms_r = syms_c.copy()
        np.random.default_rng(seed_i).shuffle(syms_r)
        c_r = cantor.copy()
        c_r.set_chemical_symbols(syms_r)
        save(c_r, "FCC HEA", "CrMnFeCoNi", f"Cantor_108_random{seed_i}", "fcc_hea")
except Exception:
    traceback.print_exc()

# ============================================================
# 9. MAX Phases
# ============================================================
print("\n== MAX Phases ==")

# --- Ti2AlC (P63/mmc #194, M2AX type) ---
try:
    # a=3.04, c=13.60 Å; Ti at 4f (1/3,2/3,~0.082), Al at 2c (1/3,2/3,0.25), C at 2a (0,0,0)
    ti2alc = crystal(
        ["Ti", "Al", "C"],
        [(1 / 3, 2 / 3, 0.082), (1 / 3, 2 / 3, 0.25), (0, 0, 0)],
        spacegroup=194, cellpar=[3.04, 3.04, 13.60, 90, 90, 120],
    )
    save(ti2alc, "MAX Phase", "Ti2AlC", "M2AX_ideal", "max_phase")
    ti2alc_s = ti2alc.repeat([2, 2, 1])
    save(ti2alc_s, "MAX Phase", "Ti2AlC", "M2AX_ideal_2x2x1", "max_phase")
    save(del_atom(ti2alc_s, "C"), "MAX Phase", "Ti2AlC", "M2AX_C_vacancy", "max_phase")
    save(del_atom(ti2alc_s, "Al"), "MAX Phase", "Ti2AlC", "M2AX_Al_vacancy", "max_phase")
except Exception:
    traceback.print_exc()

# --- Ti3AlC2 (P63/mmc #194, M3AX2 type) ---
try:
    # a=3.07, c=18.52 Å; Ti1 at 4f (1/3,2/3,0.136), Ti2 at 2a (0,0,0),
    # Al at 2b (0,0,0.25), C at 4f (1/3,2/3,0.574) — approximate
    ti3alc2 = crystal(
        ["Ti", "Ti", "Al", "C"],
        [(1 / 3, 2 / 3, 0.136), (0, 0, 0), (0, 0, 0.25), (1 / 3, 2 / 3, 0.574)],
        spacegroup=194, cellpar=[3.07, 3.07, 18.52, 90, 90, 120],
    )
    save(ti3alc2, "MAX Phase", "Ti3AlC2", "M3AX2_ideal", "max_phase")
    save(del_atom(ti3alc2, "C"), "MAX Phase", "Ti3AlC2", "M3AX2_C_vacancy", "max_phase")
    save(del_atom(ti3alc2, "Al"), "MAX Phase", "Ti3AlC2", "M3AX2_Al_vacancy", "max_phase")
except Exception:
    traceback.print_exc()

# ============================================================
# 10. Thermoelectrics
# ============================================================
print("\n== Thermoelectrics ==")

# --- PbTe rock-salt (Fm-3m #225, a=6.454 Å) ---
try:
    pbte = crystal(
        ["Pb", "Te"], [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225, cellpar=[6.454] * 3 + [90] * 3,
    )
    save(pbte, "Thermoelectric", "PbTe", "rocksalt_ideal", "thermoelectric")
    pbte_s = pbte.repeat([2, 2, 2])
    save(pbte_s, "Thermoelectric", "PbTe", "rocksalt_ideal_2x2x2", "thermoelectric")
    save(del_atom(pbte_s, "Te"), "Thermoelectric", "PbTe", "rocksalt_Te_vacancy", "thermoelectric")
    save(del_atom(pbte_s, "Pb"), "Thermoelectric", "PbTe", "rocksalt_Pb_vacancy", "thermoelectric")
    # Sn-alloyed: Pb0.75Sn0.25Te (common thermoelectric alloy)
    pbsnte = pbte_s.copy()
    pb_sites = [i for i, s in enumerate(pbsnte.get_chemical_symbols()) if s == "Pb"]
    n_sn = int(len(pb_sites) * 0.25)
    for i in np.random.default_rng(SEED).choice(pb_sites, size=n_sn, replace=False):
        syms = list(pbsnte.get_chemical_symbols()); syms[i] = "Sn"; pbsnte.set_chemical_symbols(syms)
    save(pbsnte, "Thermoelectric", "Pb0.75Sn0.25Te", "rocksalt_alloyed", "thermoelectric")
except Exception:
    traceback.print_exc()

# --- Bi2Te3 (R-3m #166, a=4.386, c=30.497 Å) ---
try:
    # Bi at 6c (0,0,0.400), Te1 at 6c (0,0,0.212), Te2 at 3a (0,0,0)
    bi2te3 = crystal(
        ["Bi", "Te", "Te"],
        [(0, 0, 0.400), (0, 0, 0.212), (0, 0, 0)],
        spacegroup=166, cellpar=[4.386, 4.386, 30.497, 90, 90, 120],
    )
    save(bi2te3, "Thermoelectric", "Bi2Te3", "rhombohedral_ideal", "thermoelectric")
    save(del_atom(bi2te3, "Te"), "Thermoelectric", "Bi2Te3", "rhombohedral_Te_vacancy", "thermoelectric")
    save(del_atom(bi2te3, "Bi"), "Thermoelectric", "Bi2Te3", "rhombohedral_Bi_vacancy", "thermoelectric")
except Exception:
    traceback.print_exc()

# ============================================================
# 11. Nuclear Materials
# ============================================================
print("\n== Nuclear Materials ==")

# --- UO2 fluorite (Fm-3m #225, a=5.468 Å) — same prototype as CeO2 ---
try:
    uo2 = crystal(
        ["U", "O"], [(0, 0, 0), (0.25, 0.25, 0.25)],
        spacegroup=225, cellpar=[5.468] * 3 + [90] * 3,
    )
    save(uo2, "Nuclear Material", "UO2", "fluorite_ideal", "nuclear")
    uo2_s = uo2.repeat([2, 2, 2])
    save(uo2_s, "Nuclear Material", "UO2", "fluorite_ideal_2x2x2", "nuclear")
    save(del_atom(uo2_s, "O"), "Nuclear Material", "UO2", "fluorite_O_vacancy", "nuclear")
    save(del_atom(uo2_s, "U"), "Nuclear Material", "UO2", "fluorite_U_vacancy", "nuclear")
    # Pu-alloyed: (U0.8Pu0.2)O2 — relevant for MOX fuel
    mox = uo2_s.copy()
    u_sites = [i for i, s in enumerate(mox.get_chemical_symbols()) if s == "U"]
    n_pu = int(len(u_sites) * 0.2)
    for i in np.random.default_rng(SEED).choice(u_sites, size=n_pu, replace=False):
        syms = list(mox.get_chemical_symbols()); syms[i] = "Pu"; mox.set_chemical_symbols(syms)
    save(mox, "Nuclear Material", "U0.8Pu0.2O2", "MOX_fluorite", "nuclear")
except Exception:
    traceback.print_exc()

# --- ZrO2 monoclinic (P21/c #14, a=5.15, b=5.21, c=5.32 Å, beta=99.2°) ---
try:
    zro2 = crystal(
        ["Zr", "O", "O"],
        [(0.2758, 0.0411, 0.2089), (0.0703, 0.3359, 0.3423), (0.4496, 0.7549, 0.4789)],
        spacegroup=14, cellpar=[5.15, 5.21, 5.32, 90, 99.2, 90],
    )
    save(zro2, "Nuclear Material", "ZrO2", "monoclinic_ideal", "nuclear")
    zro2_s = zro2.repeat([2, 2, 2])
    save(zro2_s, "Nuclear Material", "ZrO2", "monoclinic_ideal_2x2x2", "nuclear")
    save(del_atom(zro2_s, "O"), "Nuclear Material", "ZrO2", "monoclinic_O_vacancy", "nuclear")
    save(del_atom(zro2_s, "Zr"), "Nuclear Material", "ZrO2", "monoclinic_Zr_vacancy", "nuclear")
    # Y-stabilized ZrO2 (YSZ): ~8 mol% Y2O3 → replace ~16% Zr with Y
    ysz = zro2_s.copy()
    zr_sites = [i for i, s in enumerate(ysz.get_chemical_symbols()) if s == "Zr"]
    n_y = max(1, int(len(zr_sites) * 0.16))
    for i in np.random.default_rng(SEED).choice(zr_sites, size=n_y, replace=False):
        syms = list(ysz.get_chemical_symbols()); syms[i] = "Y"; ysz.set_chemical_symbols(syms)
    save(ysz, "Nuclear Material", "Zr0.84Y0.16O2", "YSZ_fluorite", "nuclear")
except Exception:
    traceback.print_exc()

# ============================================================
# Write metadata CSV
# ============================================================
with open(META, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["structure_id", "material_class", "formula", "variant", "file_path"])
    writer.writerows(records)

print(f"\n{'='*60}")
print(f"Total structures generated: {len(records)}")
print(f"Metadata: {META}")
print(f"Structures: {OUT}/")
