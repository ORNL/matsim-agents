# scripts/

These are internal tools for maintainers and specific research campaigns
(NbTaO5 preparation, E1/E2 evaluation, paper figures and model benchmarks).
Their options and outputs can change without notice, and they are not part of
the supported user interface.

For supported functionality, use the `matsim-agents` command:

- `matsim-agents --help` lists all commands.
- `matsim-agents tools list` lists the package's training, dataset and benchmark tools;
  run one with `matsim-agents tools run NAME [ARGS...]`.
- `scripts/check.sh` runs the repository's lint and test checks (`lint`, `test`, `docs`).
