# Collinear magnetic examples

All commands prepare starting states only. None runs DFT, and none is a qualified result.

| Case | Command | Policy and trials |
|---|---|---|
| NM W-rich | `matsim-agents magnetic generate w_rich.extxyz -o states/w` | `refractory_w_ta_cr_v_hf_mo`: nonmagnetic |
| FM Fe | `matsim-agents magnetic generate fe.extxyz -o states/fe --policy fe_cr_rafm` | ferromagnetic, nonmagnetic (no Cr, so no antiparallel trial) |
| AFM Fe (BCC) | `matsim-agents magnetic generate fe.extxyz -o states/fe_afm --policy fe_cr_rafm --lattice-a 2.87` | add a `bcc_afm` trial through the Python API (`magnetism_generators.bcc_afm`) |
| Custom Fe-Cr | `matsim-agents magnetic generate fecr.extxyz -o states/fecr` | ferromagnetic, Cr antiparallel, nonmagnetic |
| Magnetic vacancy migration | Python: `magnetism_defects.derive_defect_configuration`, then `check_neb_magnetic_compatibility` | endpoints and images share one parent state |

Status levels: software implemented and tested; numerically validated: no; scientifically
qualified: no. `matsim-agents magnetic policies` prints the same status.
