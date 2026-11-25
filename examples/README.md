# Applications and examples

`src/matsim_agents` is a general framework. Each scientific campaign is an
application here: an `application.yaml` names the system, the workflow that
drives it and that workflow's config. The framework reads it with
`matsim_agents.core.application.load_application` and holds no campaign
constants of its own. `tests/test_package_boundaries.py` enforces this.

## Application file

| Field | Meaning |
| --- | --- |
| `name`, `description` | Identifier and summary. |
| `paradigm` | Kind of study, e.g. `supercell_alloy` or `prototype_search`. |
| `workflow` | The framework workflow that consumes the config. |
| `target` | For supercell applications: `composition`, `supercell` repeats, `parent_lattice`, `expected_sites`. Site counts must agree. |
| `compositions` | For composition-driven applications: the formulas to explore. |
| `species`, `properties` | Elements and target properties; `species` must cover the target. |
| `workflow_config` | Path, relative to the application file, of the workflow config. |
| `parameters` | Application-specific values passed to the workflow, such as transmutation host and products. |

Check any application against its workflow without computing anything:

```bash
matsim-agents application check examples/fusion_bcc_1024/application.yaml
```

```python
from matsim_agents.core.application import load_application

app = load_application("examples/fusion_bcc_1024/application.yaml")
config_path = app.resolve(app.workflow_config)
```

## Applications

| Application | Paradigm | How to run it |
| --- | --- | --- |
| [fusion_bcc_1024](fusion_bcc_1024/application.yaml) | Fixed 1,024-site BCC W640Ta128Cr64V96Hf32Mo64 supercell, (8,8,8) repeats, with transmutation and gas. | `matsim_agents.workflows.full_supercell_campaign.FullSupercellCampaign` with [campaign.yaml](fusion_bcc_1024/campaign.yaml). `tests/test_applications.py` pins the target. |
| [cantor_fcc_256](cantor_fcc_256/application.yaml) | Non-BCC (4x4x4 FCC CoCrFeMnNi, 256 sites) hierarchical transfer campaign; `application check` builds the FCC parent. | `matsim_agents.workflows.hierarchy.engine` with [campaign.yaml](cantor_fcc_256/campaign.yaml); see [hierarchical-transfer-campaign.md](../docs/hierarchical-transfer-campaign.md). |
| [perovskite_oxides](perovskite_oxides/application.yaml) | Composition-driven perovskite prototype search. | `python examples/perovskite_oxides/run.py -o perovskite-candidates` generates prototype candidates; relaxation needs an MLIP config. |

The other directories hold configs and scripts for single framework features:
active learning, debate, evaluation, characterization, MCP and paper cases.
