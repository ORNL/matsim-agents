# Fusion plasma-facing materials: eight-objective scoring policy

Status: implemented and tested with SYNTHETIC data only. No benchmark anchor in the
repository is a material property, and no score is a materials claim.

## Objectives

`matsim_agents.science.decision.policy.FUSION_PFC_OBJECTIVES` lists the eight objectives. Each is
maximized as a desirability $d_i \in [0, 1]$:

| Objective | Scored observables (see `matsim_agents.science.decision.objectives`) |
|---|---|
| radiation_damage_recovery | defect recovery fraction |
| thermal_conductivity | total thermal conductivity |
| mechanical_integrity | yield strength, fracture toughness (min over both) |
| thermal_shock_fatigue | thermal-shock damage threshold |
| plasma_erosion_resistance | sputtering yield (plasma ion, energy, incidence angle) |
| high_temperature_stability | maximum phase-stable temperature |
| low_hydrogen_retention | per-isotope retention (H, D, T kept separate) |
| low_activation_shutdown_dose | shutdown dose rate (dose model and geometry) |

The registry declares which observables are required, which are insufficient alone,
and which conditions must match before evidence is used.

## Score

For a candidate with every objective present,

$$S = \sum_i w_i d_i, \qquad d_i = \min_{o \in i} \mathrm{clip}\left(\frac{x_o - x_o^{\mathrm{worst}}}{x_o^{\mathrm{target}} - x_o^{\mathrm{worst}}}, 0, 1\right)$$

The weights come from a versioned `FusionScoringPolicy`, and `digest()` ties every
result to its policy. `[S_low, S_high]` propagates the one-sigma bounds of each
observable.

- **Incomplete candidates** get no score. A missing objective is never zero-filled.
  The candidate is reported as `incomplete`, with its partial weighted sum and the
  weight it covers.
- **Constraints** use only evidence whose conditions match the scenario fields of the
  constrained property. A violated constraint removes the candidate from the ranking.
- **Pareto and dominance** are computed on complete objectives. When some candidates
  are incomplete, the front is marked partial.
- **Sensitivity** scales each weight and reports how the ranking changes.
- **Scenarios**: `matsim_agents.science.decision.scenario.aggregate` normalizes per-scenario
  scores to the declared scenario ids, and refuses scores for undeclared ones.

## Command line

`matsim-agents materials validate|score|report|pareto|sensitivity|packet-export|import`.
Use `--synthetic` for the A/B/C demo bundle, or pass a YAML bundle with `scenario`,
`candidates`, optional `constraints` and `benchmarks`. Every command reads evidence
only. The SYNTHETIC end-to-end demo is
[examples/fusion_pfc/synthetic_demo.py](../../examples/fusion_pfc/synthetic_demo.py).

## Acquisition advice

`matsim_agents.science.decision.acquisition.plan_fusion_acquisition` ranks proposed calculations
by expected gain in decision confidence per node-hour. Decision confidence is
$P(\text{top}) \cdot n_\text{complete}/n_\text{ranked}$.

- **Unquantified**: a proposal with no current value and no labelled prior. Its gain
  is not invented.
- **Refused**: proxies.
- **Budget**: selection stays within the real property budget minus the validation
  reserve.
- **Approval**: plans dispatch nothing. `reserve_selected` needs a named approver.

Tests: [tests/test_fusion_policy.py](../../tests/test_fusion_policy.py),
[tests/test_fusion_scoring.py](../../tests/test_fusion_scoring.py),
[tests/test_fusion_acquisition.py](../../tests/test_fusion_acquisition.py),
[tests/test_fusion_cli.py](../../tests/test_fusion_cli.py).
