# Hypothesis-driven discovery chat

## Hypothesis-driven discovery chat

The `chat` REPL is more than a wrapper around the LLM — it is a
**closed loop between dialogue and atomistic simulation**:

1. The user and the assistant exchange messages about a target property.
2. After each turn, [`extract_compositions`](../../src/matsim_agents/discovery/composition.py) scans both messages for chemical formulas (validates element symbols, reduces stoichiometry, ignores English words like "Carbon" or "Hello").
3. For every newly-seen formula the user is asked (or `--auto-confirm` is honored) whether to launch a substantial atomistic exploration.
4. The wrapper [`explore_composition`](../../src/matsim_agents/discovery/wrapper.py) then:
   - **generates seeds** through the unified
     [`matsim_agents.discovery.seeds.generate_seeds`](../../src/matsim_agents/discovery/seeds.py)
     entry point, which combines:
     1. **AFLOW prototype decoration.** Every prototype in the
        pymatgen-bundled AFLOW encyclopedia whose reduced stoichiometric
        ratios match the target composition is substituted with the
        target's elements. All symmetrically distinct
        element-to-placeholder assignments are enumerated (e.g. ABX₃
        vs BAX₃). This recovers fcc/bcc/hcp/rocksalt/zincblende/
        wurtzite/fluorite/rutile/perovskite/spinel/Heusler/MAX/… from a
        single uniform source — no per-stoichiometry rules.
     2. **pyXtal random search** (`--n-random N`). `N` random crystals
        are drawn uniformly across the 230 space groups, respecting
        Wyckoff multiplicities and minimum interatomic distances. Each
        such seed is tagged `needs_dft_verification=True` and surfaced
        with a `(novel)` marker in the live table so it is treated as
        a candidate for follow-up DFT validation rather than a
        publishable claim. When the target composition has no AFLOW
        match (e.g. a 5-element high-entropy alloy), this is the only
        active source and `--n-random` should be raised accordingly.
   - **relaxes** each seed with HydraGNN + ASE (FIRE/BFGS).
   - **scores** chemical stability (ΔE/atom ranking, near-degeneracy
     warning) and a dynamical-stability proxy (max residual force),
     keeping the `source` (`prototype` vs `random`) and AFLOW
     `prototype_id` / `space_group` of every candidate in the report.
5. The summary is fed back into the conversation as a discovery user-turn
  payload so the LLM can refine its hypothesis on the next turn.

Discovery chat can also run optional control actions through slash commands:

- **Single-structure relaxation command**:
  - `/relax path/to/structure.vasp`
- **Active-learning handoff command**:
  - `/al` (use the most recently discussed composition)
  - `/al Li2MnO3` (specify the composition explicitly)
  - Requires `--al-config`; respects the `--al-dry-run/--al-run` setting.
- **Reset command**:
  - `/clear` (clear the conversation history and discovery state for the session)
- **UQ-based AL handoff** (automatic policy knobs on CLI):
  - `--trigger-al-handoff/--no-trigger-al-handoff`
  - `--al-config <base_al_yaml>`
  - `--al-dry-run/--al-run`
  - `--uq-top-weight-threshold`
  - `--uq-min-unreliable-fraction`
  - `--uq-min-relaxations-for-handoff`
  - `--al-handoff-audit-path`

If `--al-handoff-audit-path` is not set, handoff events default to:

```
<output_dir>/discovery/al_handoff_events.jsonl
```

Every session writes its transcript as it goes, to
`<output_dir>/chat/<UTC timestamp>_<suffix>/`: `transcript.md` for reading,
`contributions.jsonl` with one record per message (your turns, the proposer's
replies, critiques and revisions when the hypothesis debate is on, and command
and exploration results) including each model's reasoning, finish reason, and
token counts, and `prompts/<id>.json` with the messages sent for each model
reply. The proposer and critics see each other's answers only, never their
reasoning. After every turn the session state (conversation, compositions
already explored, and their results) is saved atomically to `session.json` in
the same directory; `matsim-agents chat --resume <session id or directory>`
continues it, appending to the same transcript and skipping compositions that
were already explored. Models and tools come from the new command line.

Output artifacts per composition (under `--output-dir`):

```
outputs/discovery/<formula>/
  seeds/    <formula>_<prototype_id>[_v<k>].vasp          # AFLOW decoration variant k
            <formula>_random_<sg>_<i>.vasp               # pyXtal seed in space group <sg>
  relaxed/  <formula>_<seed>_optimized_structure.vasp
            <formula>_<seed>_optimization.traj           # ASE trajectory
            <formula>_<seed>_optimization.csv            # per-step E, |F|max, branch weights
```

Seeds carry their provenance (`source`, `prototype_id`, `space_group`,
`needs_dft_verification`) on the `PhaseCandidate` Pydantic model so
downstream scorers can filter or weight them.

> **Honest caveats.** The AFLOW prototype set covers known crystal
> topologies for stoichiometries that match an existing entry — exotic
> ratios fall back to the pyXtal random pass, which is novelty-oriented
> and intentionally flagged for DFT verification. The dynamical-
> stability check is a force-residual proxy, not a full phonon
> analysis; plug in phonopy for the rigorous version. For broader
> generative coverage (CALYPSO, USPEX, AIRSS, diffusion models, …) add
> a new branch to `generate_seeds` — every consumer already routes
> through that single entry point.
>
> **Composition detection is regex-based.** The chat REPL extracts
> target materials by pattern-matching chemical formulas (`Li2MnO3`,
> `CrMoNbTaW`) in user and assistant text via
> [`extract_compositions`](../../src/matsim_agents/discovery/composition.py).
> This is intentionally minimal but has known failure modes:
> - **Fires on context, not intent.** `"CO2 emissions are dominated by
>   CaO formation"`, `"Following Smith et al. on BaTiO3 ferroelectrics
>   we instead study SrTiO3"`, or `"avoid the toxic As2O3 phase"` will
>   all trigger a prompt for the wrong (or every) formula. Space-group
>   strings like `"P3"`, `"C2/c"`, and DFT-functional names like
>   `"B3LYP"` parse as `P+3`, `C+2`, `B+3` and pass the validator.
> - **Misses verbal proposals.** `"lithium manganate at the 2-1-3
>   stoichiometry"`, `"the Li-Mn-O ternary"`, Unicode subscripts
>   (`Li₂MnO₃`), parentheses-with-alternation (`(Li,Na)2MnO3`), and
>   fractional stoichiometries (`Li2Mn0.5Ni0.5O2`) are not detected.
> - **Cannot read polarity.** `"DO NOT explore Li2MnO3"` triggers the
>   same prompt as `"please explore Li2MnO3"`. The interactive y/N
>   confirmation (or `--auto-confirm` for batch runs) is what stands
>   between these and a wasted multi-hour relaxation.
>
> The natural replacement is a tool-calling LLM that explicitly invokes
> `explore_composition(formula, rationale)` when it actually means to
> compute the material, which would eliminate every class above; a
> migration of the chat REPL onto a LangGraph `ToolNode` is the right
> moment to do this.

---
