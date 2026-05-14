"""Programmatic example: run the agent graph on a single structure."""

from __future__ import annotations

import uuid

from matsim_agents.graph import build_graph
from matsim_agents.state import MatSimState


def main() -> None:
    objective = (
        "Relax the structure at structures/mos2-B_Defect-Free_PBE.vasp using HydraGNN "
        "and report the final energy."
    )
    graph = build_graph()
    final = graph.invoke(
        MatSimState(objective=objective),
        config={
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "logdir": "multidataset_hpo-BEST6-fp64",
                "mlp_checkpoint": "mlp_branch_weights.pt",
                "mlp_device": "cuda",
            }
        },
    )
    print(final.get("analysis"))


if __name__ == "__main__":
    main()
