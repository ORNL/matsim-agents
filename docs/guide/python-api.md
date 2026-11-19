# Python API

## Programmatic API

### Single relaxation

```python
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip

result = relax_with_mlip(RelaxStructureInput(
    structure_path="structures/mos2.vasp",
    logdir="./multidataset_hpo-BEST6-fp64",
  hydragnn_branch_mlp_checkpoint="./mlp_branch_weights.pt",
    optimizer="FIRE",
    maxiter=200,
))
print(result.final_energy_eV, result.optimized_structure_path)
```

### Composition exploration

```python
from matsim_agents.discovery import explore_composition

# Default: every applicable AFLOW prototype + 50 pyXtal random seeds.
result = explore_composition(
    "Cs2AgBiBr6",
    logdir="./multidataset_hpo-BEST6-fp64",
  hydragnn_branch_mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
)
print(result.stability.summary)

# Prototype-only run (pyXtal pass disabled).
result = explore_composition(
    "MoS2",
    logdir="./multidataset_hpo-BEST6-fp64",
  hydragnn_branch_mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
    n_random=0,
)

# Novelty-heavy run for an exotic / high-entropy composition with no
# AFLOW match — rely entirely on pyXtal.
result = explore_composition(
    "FeCoNiCrMn",
    logdir="./multidataset_hpo-BEST6-fp64",
  hydragnn_branch_mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
    n_random=200,
    random_seed=42,
)
```

### Run the core LangGraph workflow

```python
import uuid
from matsim_agents.orchestration.objective_graph import build_graph
from matsim_agents.orchestration.state import MatSimState

graph = build_graph()
final = graph.invoke(
    MatSimState(
        objective="Relax structures/foo.vasp and summarize.",
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    ),
    config={"configurable": {
        "thread_id": str(uuid.uuid4()),
        "logdir": "./multidataset_hpo-BEST6-fp64",
      "hydragnn_branch_mlp_checkpoint": "./mlp_branch_weights.pt",
    }},
)
print(final["analysis"])
```

### Run the supervisor LangGraph workflow

```python
from matsim_agents.orchestration.composition_graph import SupervisorConfig, run_supervisor

final = run_supervisor(SupervisorConfig(
  composition="Li2MnO3",
  logdir="./multidataset_hpo-BEST6-fp64",
  hydragnn_branch_mlp_checkpoint="./mlp_branch_weights.pt",
  output_dir="./outputs",
  trigger_active_learning_on_high_uq=True,
  active_learning_config="examples/active_learning/al_config.example.yaml",
  active_learning_dry_run=True,
))
print(final.get("summary"))
```

### Embed the chat loop in your own app

```python
from matsim_agents.orchestration.interactive_chat import DiscoveryChatConfig, DiscoveryChatSession, chat_once

session = DiscoveryChatSession(config=DiscoveryChatConfig(
    logdir="./multidataset_hpo-BEST6-fp64",
  hydragnn_branch_mlp_checkpoint="./mlp_branch_weights.pt",
    output_dir="./outputs",
    llm_model="qwen2.5:14b",
    auto_confirm=True,
))
reply = chat_once(session, "Propose a Pb-free perovskite for PV.")
```

---
