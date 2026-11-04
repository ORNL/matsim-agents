# Alloy DFT settings: adequacy for E2 relative-energy questions

Applies to attempt `e2-dft-v1-20261104` (Aurora job 9040206). Cases: Cantor CrMnFeCoNi (108 atoms) and NbTaVHfZrTi (128 atoms), each as "ideal" plus random1–3.

## What the four configurations are

- All eight POSCARs are random-like chemical arrangements on undistorted lattices. FCC has CN1 = 12; BCC has CN1 = 8.
- First-shell Warren–Cowley |α| means:
  - FCC: ideal 0.078; random1–3 0.077–0.095.
  - BCC: ideal 0.116; random1–3 0.107–0.140.
- The archive label "ideal" therefore marks one more random realization, not an ordered or SQS reference.
- Valid question: the spread in configurational energy among four realizations.
- Invalid question: "ideal vs random" stability.

## Settings and limitations

| Setting | Value | Assessment |
|---|---|---|
| Cutoff | FCC 480 eV, BCC 345 eV (1.3 × max ENMAX) | Adequate for ISIF=3 at fixed basis. Every configuration uses the same cutoff and PAW set, so basis errors largely cancel in differences between configurations. Absolute volumes carry the usual Pulay-stress bias, which has not been quantified. |
| k-mesh | 2×2×2 Γ (FCC 864 KPPRA, 0.29 Å⁻¹; BCC 1024 KPPRA, 0.24 Å⁻¹), MP1 σ = 0.1 eV | Coarse for metals. Absolute energies may carry k-sampling errors of several meV/atom. Using the same cell, mesh and smearing cancels part of the error, but the residual error on configuration differences is **not measured**. Configurational spreads in 100-atom HEA cells are typically a few meV/atom, the same order. Rankings below about 5 meV/atom must therefore stay unresolved until checked. |
| Cantor magnetism | Collinear, ferromagnetic initialization (Co 2, Cr 3, Fe 3, Mn 3, Ni 1 μB) | Cr and Mn favour antiparallel or frustrated local moments, so a ferromagnetic start can reach different magnetic local minima in different configurations. Magnetic-state differences can exceed chemical-ordering differences. The final moments must be reported per configuration, and any ranking is conditional on the magnetic state found. Paramagnetic (DLM or spin-disordered) physics is not represented. |
| BCC magnetism | ISPIN = 1 | Standard for refractory HEAs, which are non-magnetic experimentally. However, the unconverged archive SCF showed 28–31 μB per cell. Non-magnetism is assumed, not verified, for this PAW set. |
| Relaxation criterion | \|F\| ≤ 0.02 eV/Å, ISIF = 3 | Adequate for meV/atom energy differences. Residual stress is reported but is not an acceptance criterion. |

## Minimal targeted checks (proposed, not run, need authorization)

1. **k-sampling:** 3×3×3 Γ static on the relaxed ideal and random1 cells for both alloys (4 statics). This measures the k-sampling error on one configuration difference.
2. **Magnetism:** for Cantor, one static on the relaxed random1 with a randomized-sign initial MAGMOM. For BCC, one ISPIN = 2 static on the relaxed ideal cell.

These are 6 single-point calculations, not a convergence sweep. They run only if the relaxed spread falls within the 5 meV/atom ambiguity band.
