"""Generate prototype candidates for every composition of the perovskite application.

Structures are generated and written, nothing is relaxed. Relaxation and hull ranking use
:func:`matsim_agents.workflows.composition_exploration.explore_composition` with an
explicit ``MLIPConfig``.

    python examples/perovskite_oxides/run.py -o perovskite-candidates
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matsim_agents.core.application import load_application
from matsim_agents.workflows.composition_exploration import generate_composition_candidates

APPLICATION = Path(__file__).with_name("application.yaml")


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    app = load_application(APPLICATION)
    counts: dict[str, int] = {}
    for formula in app.compositions:
        _, candidates, _ = generate_composition_candidates(
            formula,
            str(args.output_dir / formula),
            n_random=int(app.parameters.get("n_random", 0)),
            random_seed=int(app.parameters.get("random_seed", 0)),
        )
        counts[formula] = len(candidates)
        print(f"{formula}: {len(candidates)} candidates")
    return counts


if __name__ == "__main__":
    main()
