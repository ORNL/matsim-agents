# Python API

Every entry point takes the MLIP as `mlip=`: a `HydraGNNConfig`, `UMAConfig`, or
`MACEConfig` (from `matsim_agents.backends.mlip.config`), or an equivalent
mapping with a `backend` key. `load_mlip_config(path)` reads the `mlip:` block of
a YAML file such as `examples/mlip/uma-s-1p1-omat.yaml`, and
`parse_mlip_config(mapping)` validates a mapping. The examples below use a
HydraGNN model; any of the three configurations works:

```python
from matsim_agents.backends.mlip.config import HydraGNNConfig

mlip = HydraGNNConfig(
    logdir="./multidataset_hpo-BEST6-fp64",
    mlp_checkpoint="./mlp_branch_weights.pt",
)
```

### Single relaxation

```python
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip

result = relax_with_mlip(RelaxStructureInput(
    structure_path="examples/characterization/structures/MgO-rocksalt.vasp",
    mlip=mlip,
    optimizer="FIRE",
    maxiter=200,
))
print(result.final_energy_eV, result.optimized_structure_path)
```

### Composition exploration

```python
from matsim_agents.discovery import explore_composition

# Default: every applicable AFLOW prototype + 50 pyXtal random structures.
# The seed is always explicit (docs/reproducibility.md).
result = explore_composition(
    "Cs2AgBiBr6",
    mlip=mlip,
    output_dir="./outputs",
    random_seed=20261113,
)
print(result.stability.summary)

# Prototype-only run (pyXtal pass disabled).
result = explore_composition(
    "MoS2",
    mlip=mlip,
    output_dir="./outputs",
    n_random=0,
    random_seed=None,
)

# Random-enriched run for an exotic / high-entropy composition. Matching
# AFLOW prototypes are still included; with none, only pyXtal contributes.
# Fewer than n_random structures may be generated.
result = explore_composition(
    "FeCoNiCrMn",
    mlip=mlip,
    output_dir="./outputs",
    n_random=200,
    random_seed=42,
)
```

### Run the core LangGraph workflow

```python
import uuid
from matsim_agents.agents.objective_graph import build_graph
from matsim_agents.agents.state import MatSimState

graph = build_graph()
final = graph.invoke(
    MatSimState(
        objective="Relax examples/characterization/structures/MgO-rocksalt.vasp and summarize.",
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    ),
    config={"configurable": {
        "thread_id": str(uuid.uuid4()),
        "mlip": mlip.model_dump(mode="json"),
    }},
)
print(final["analysis"])
```

### Run the supervisor LangGraph workflow

```python
from matsim_agents.agents.composition_graph import SupervisorConfig, run_supervisor

final = run_supervisor(SupervisorConfig(
  composition="Li2MnO3",
  mlip=mlip,
  output_dir="./outputs",
  trigger_active_learning_on_high_uq=True,
  active_learning_config="examples/active_learning/al_config.example.yaml",
  active_learning_dry_run=True,
))
print(final.get("summary"))
```

### Embed the chat loop in your own app

```python
from matsim_agents.agents.interactive_chat import DiscoveryChatConfig, DiscoveryChatSession, chat_once

session = DiscoveryChatSession(config=DiscoveryChatConfig(
    mlip=mlip,
    output_dir="./outputs",
    llm_model="qwen2.5:14b",
    auto_confirm=True,
))
reply = chat_once(session, "Propose a Pb-free perovskite for PV.")
```

---
