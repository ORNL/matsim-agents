# E3-MLIP hypothesis-evolution evidence

This index accompanies the saved acquisition and outcome artifacts for the
`e3-mlip-production` cohort, `llm_guided_adaptation` strategy, seed `20261113`.
The acquisition proposals record the prompt, model response, accepted allocation,
and preceding strategy-local outcome summaries. Round summaries record outcomes;
they are not themselves final LLM interpretations. The original files are
preserved at their run paths.

Read the [chronological audit and complete verbatim rationales](./hypothesis-evolution-audit.md)
for the recorded evolution, numerical feedback, sampling decisions, and gaps.

| Material | Round 1 proposal | Round 2 proposal | Round 3 proposal | Campaign summary |
|---|---|---|---|---|
| BCC refractory | [R1](../../../runs/e3-mlip-production/bcc_refractory/llm_guided_adaptation/seed-20261113/round-01/acquisition/proposal-001.json) | [R2](../../../runs/e3-mlip-production/bcc_refractory/llm_guided_adaptation/seed-20261113/round-02/acquisition/proposal-001.json) | [R3](../../../runs/e3-mlip-production/bcc_refractory/llm_guided_adaptation/seed-20261113/round-03/acquisition/proposal-001.json) | [summary](../../../runs/e3-mlip-production/bcc_refractory/llm_guided_adaptation/seed-20261113/campaign-summary.json) |
| FCC Cantor | [R1](../../../runs/e3-mlip-production/fcc_cantor/llm_guided_adaptation/seed-20261113/round-01/acquisition/proposal-001.json) | [R2](../../../runs/e3-mlip-production/fcc_cantor/llm_guided_adaptation/seed-20261113/round-02/acquisition/proposal-001.json) | [R3](../../../runs/e3-mlip-production/fcc_cantor/llm_guided_adaptation/seed-20261113/round-03/acquisition/proposal-001.json) | [summary](../../../runs/e3-mlip-production/fcc_cantor/llm_guided_adaptation/seed-20261113/campaign-summary.json) |
| LiFePO4 | [R1](../../../runs/e3-mlip-production/lifepo4/llm_guided_adaptation/seed-20261113/round-01/acquisition/proposal-001.json) | [R2](../../../runs/e3-mlip-production/lifepo4/llm_guided_adaptation/seed-20261113/round-02/acquisition/proposal-001.json) | [R3](../../../runs/e3-mlip-production/lifepo4/llm_guided_adaptation/seed-20261113/round-03/acquisition/proposal-001.json) | [summary](../../../runs/e3-mlip-production/lifepo4/llm_guided_adaptation/seed-20261113/campaign-summary.json) |
| Phosphorene | [R1](../../../runs/e3-mlip-production/phosphorene/llm_guided_adaptation/seed-20261113/round-01/acquisition/proposal-001.json) | [R2](../../../runs/e3-mlip-production/phosphorene/llm_guided_adaptation/seed-20261113/round-02/acquisition/proposal-001.json) | [R3](../../../runs/e3-mlip-production/phosphorene/llm_guided_adaptation/seed-20261113/round-03/acquisition/proposal-001.json) | [summary](../../../runs/e3-mlip-production/phosphorene/llm_guided_adaptation/seed-20261113/campaign-summary.json) |
| Nb-Ta-O | [R1](../../../runs/e3-mlip-production/nb_ta_o/llm_guided_adaptation/seed-20261113/round-01/acquisition/proposal-001.json) | [R2](../../../runs/e3-mlip-production/nb_ta_o/llm_guided_adaptation/seed-20261113/round-02/acquisition/proposal-001.json) | [R3](../../../runs/e3-mlip-production/nb_ta_o/llm_guided_adaptation/seed-20261113/round-03/acquisition/proposal-001.json) | [summary](../../../runs/e3-mlip-production/nb_ta_o/llm_guided_adaptation/seed-20261113/campaign-summary.json) |
| Zn-formate MOF | [R1](../../../runs/e3-mlip-production/zn_formate_mof/llm_guided_adaptation/seed-20261113/round-01/acquisition/proposal-001.json) | [R2](../../../runs/e3-mlip-production/zn_formate_mof/llm_guided_adaptation/seed-20261113/round-02/acquisition/proposal-001.json) | [R3](../../../runs/e3-mlip-production/zn_formate_mof/llm_guided_adaptation/seed-20261113/round-03/acquisition/proposal-001.json) | [summary](../../../runs/e3-mlip-production/zn_formate_mof/llm_guided_adaptation/seed-20261113/campaign-summary.json) |

For each completed round, `round-NN/round-summary.json` beside the acquisition
directory records the round's outcome. Round 1 has no prior outcome history;
rounds 2 and 3 include cumulative prior action summaries in their prompts.
These summaries do not provide individual optimizer force or step histories.

## Interpretation and caveats

- The saved responses are acquisition rationales, not necessarily scientific
  hypotheses. Distinguish changes in material interpretation from changes in
  computational sampling. No separate post-final-round LLM interpretation was
  found in the audited artifacts.
- Action-level mean MACE energies, distinct geometry hashes, and convergence
  counts do not establish thermodynamic stability or DFT-level accuracy.
- Phosphorene's round-3 proposal was accepted but not executed: generation
  stopped after eight partial structures, with zero optimizations and no
  hypothesis update. See its [stopped-round outcome](../../../runs/e3-mlip-production/phosphorene/llm_guided_adaptation/seed-20261113/round-03/acquisition/outcome.json).
- The FCC round-3 rationale describes Cr2MnFeCoNi tier-16 convergence as
  3/11, whereas its saved prior outcome rows total 4/11. The Nb-Ta-O round-2
  rationale states 0/24 for tiers 16 and 32, whereas the prompt totals 0/26.
  Treat these as model statements rather than corrected measurements.
- The matched Zn-formate campaign reports 0/192 converged with a 300-second
  optimization limit. The [separate follow-up campaign](../../../runs/e3-mlip-zn-formate-10000-20261113/zn_formate_mof/llm_guided_adaptation/seed-20261113/campaign-summary.json)
  reports 138/192 converged with a 1,800-second limit; do not merge its
  results into the matched cohort.
