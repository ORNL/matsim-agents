# Run artifacts, provenance, and restarts

Scientific workflows write collision-resistant run directories rather than
placing mutable outputs directly in a shared working directory.

```text
runs/<UTC-timestamp>_<random-suffix>/
├── request.json
├── resolved_config.json
├── provenance.json
├── events.jsonl
├── result.json
├── structures/
├── calculations/
├── datasets/
└── models/
```

The timestamp makes runs readable to humans; the random suffix prevents two
concurrent jobs started in the same second from colliding.

## Authoritative records

- `request.json` stores what the caller requested.
- `resolved_config.json` stores defaults and environment substitutions after
  validation. It is the configuration that was actually executed.
- `provenance.json` records workflow and backend identity, software/model
  versions, checkpoint hashes, random seed, units, numerical settings, energy
  reference, and parent run or dataset identifiers.
- `events.jsonl` is an append-only state transition and decision stream.
- `result.json` contains status, evidence, convergence, validations, metrics,
  artifacts, and a failure reason when applicable.

Named subdirectories contain the immutable scientific artifacts referenced by
these records. User-facing analysis should follow recorded artifact paths
rather than guessing filenames.

## Restart contract

Active-learning runs may resume from completed iteration state. Restart logic
must:

1. recognize completed labels and never append them twice;
2. preserve the original backend and energy reference;
3. reuse validated artifacts rather than rerunning them silently;
4. create an explicit event when work is retried or skipped;
5. retain failed calculations and their reasons;
6. never promote a newly trained model without the configured approval.

A configuration or dataset change should create a new run or a new versioned
artifact with parent lineage. It must not rewrite the provenance of an older
result.

## Dataset governance

Before labeled structures enter a training dataset, validation checks finite
energies and forces, force dimensions, geometry identity, backend identity,
and duplicate hashes. The manifest records the accepted and rejected counts
and a SHA-256 digest. Files loaded through pickle-based scientific formats must
be treated as trusted inputs because Python pickle deserialization can execute
code.

## Comparing runs

Cross-machine comparisons first require the same Git commit, input structure
digest, scientific configuration, model identity, and energy reference. Only
then should energies, forces, positions, convergence, throughput, and scaling
be compared. Floating-point portability uses documented tolerances rather than
bitwise equality. See the
[portability benchmark](../benchmarks/portability/README.md).

