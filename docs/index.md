# matsim-agents documentation

Pages are grouped by what you want to do. Project plans, campaign reports and
proposals are kept separately in [project/](project/) and are not needed to use
the framework.

## 1. Install and check your environment

- [Installation](../README.md#installation)
- [Command-line guide](guide/cli.md): diagnostics, starter configurations, and agent command options
- Run `matsim-agents doctor` to see which Python, DFT, MLIP and LLM backends are usable here.
- [Backend capabilities](capabilities.md): what each backend needs and supports
- [MACE dependencies](mace-dependencies.md) and [frozen HydraGNN source](hydragnn-frozen-source.md)

## 2. Run your first calculation

- [A small phase search without writing Python](workflows/simple-exploration.md)
- [Quick start](../README.md#quick-start)
- Create a starter config with `matsim-agents config init KIND FILE`; see `matsim-agents config list`.

## 3. Choose and run a scientific workflow

- [Choosing a scientific workflow](guide/workflows.md)
- [Run a workflow or E3 campaign](workflows/run-campaigns.md)
- [Scientific workflow contracts](scientific-workflows.md)
- [Radiation-damage extension status](radiation-damage-status.md)
- [External SPECTRA-PKA interface](external/spectra-pka.md)
- [SPECTRA-PKA nuclear data (TENDL-2023)](external/spectra-pka-data.md)
- [OpenMC activation and transmutation](external/openmc.md)
- [Hierarchical transfer campaign (W-Ta-Cr-V-Hf-Mo)](hierarchical-transfer-campaign.md)
- [Characterization contracts](characterization.md)
- [Multi-property characterization, selection, and hypothesis revision](multiproperty-selection.md)
- [Scientific hypothesis debate](scientific-debate.md)
- [Hypothesis-driven discovery chat](guide/discovery-chat.md)
- [Active learning (MLIP and DFT)](guide/active-learning.md)
- [MLIP training references and model promotion](mlip-training-and-promotion.md)
- [MLIP and DFT energy conventions](mlip-energy-conventions.md): labels, references and units for each MLIP
- [Competing-phase reference hulls](reference-hulls.md)

## 4. Read results and restart safely

- [Interpret workflow results and recover safely](workflows/interpret-results.md)
- [Run artifacts, provenance, and restarts](run-artifacts-and-restarts.md)
- [Reproducibility and random seeds](reproducibility.md)
- [Evidence ledger](evidence-ledger.md)
- [Evaluating fine-tuned MLIPs](mlip-evaluation.md)

## 5. Configure LLMs

- [LLM backends](guide/llm-backends.md)
- [LLM readiness and portability qualification](llm-readiness.md)
- [vLLM versus HuggingFace Transformers](llm-backends-comparison.md)
- [Downloading model weights](model-download.md) and [download safety policy](model-download-safety.md)
- [Model Access Gateway](model-access-gateway.md)
- [External MCP tools in `matsim-agents run`](mcp.md)

## 6. Run on HPC systems

- [Running on DOE supercomputers](guide/hpc.md)
- [HPC platforms index](hpc-platforms.md) and [HPC validation matrix](hpc-validation.md)
- [Resource dispatch for LLMs, MLIPs, and DFT](resource-dispatch.md)
- [Running jobs through the DOE IRI Facility API](iri.md)
- [Distributed DFT dispatch](distributed-dft-dispatch.md)
- VASP: [installations and runtime](vasp-hpc-runtime.md), [Aurora](vasp-aurora.md),
  [Perlmutter](vasp-perlmutter.md)
- Quantum ESPRESSO: [Aurora](quantum-espresso-aurora.md),
  [Frontier](quantum-espresso-frontier.md), [Perlmutter](quantum-espresso-perlmutter.md)
- [vLLM on Aurora](vllm-aurora.md)
- [HPC task coordination](hpc-task-coordination.md) and
  [Aurora task-board read check](aurora-task-board-check.md)

## 7. Use and extend the Python library

- [Python API](guide/python-api.md)
- [Extend a workflow](workflows/extend-library.md)
- [Configuration reference](guide/configuration.md), including the `mlip:` block
- [Architecture and capabilities](guide/architecture.md) and
  [package architecture](architecture.md)
- Restructuring in progress: [architectural audit](restructuring/audit.md)
  and [target architecture](restructuring/target-architecture.md)
- Workflow diagrams: [page](workflows/agentic-ai-workflow.md),
  [slides](workflows/agentic-ai-workflow-slides.md)

## 8. Examples and benchmarks

- Example configurations: [MLIP selection](../examples/mlip/),
  [active learning](../examples/active_learning/README.md),
  [characterization](../examples/characterization/),
  [debate](../examples/debate/), [workflows](../examples/workflows/), and
  [paper cases](../examples/paper_cases/README.md)
- [Codabench competition](guide/codabench.md)
- [Benchmarks overview](../benchmarks/README.md),
  [Si/SiC known-answer suite](../benchmarks/known_answers/README.md), and
  [portability release gate](../benchmarks/portability/README.md)
