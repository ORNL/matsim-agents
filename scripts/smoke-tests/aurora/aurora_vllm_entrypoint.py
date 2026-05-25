#!/usr/bin/env python3
"""Aurora vLLM entrypoint.

Thin wrapper that patches vLLM's ``_run_in_subprocess`` *before* the API
server starts, so that the hook can be customised without modifying vLLM.

On Aurora (frameworks/2025.3.1) the registry subprocess works correctly
without any environment overrides: a plain ``subprocess.run()`` child
inherits PALS device-fabric permissions from the mpiexec-launched API server
process.  The patch currently just replicates the default behaviour; it
exists as an extensibility point in case node-specific adjustments are
needed (e.g. setting ``ONEAPI_DEVICE_SELECTOR=cpu:*`` to skip Level Zero
initialisation if a future driver regression re-introduces SIGSEGV in the
inspection subprocess).

Usage::

    mpiexec -n 1 --ppn 1 \\
      env -u PMI_RANK ... \\
      python aurora_vllm_entrypoint.py [api_server_args...]

All CLI arguments after the script name are forwarded unchanged to the API
server via ``sys.argv``.
"""
import os
import pickle
import subprocess
import sys
import tempfile


def _patch_registry_subprocess() -> None:
    """Replace ``_run_in_subprocess`` in vllm.model_executor.models.registry."""
    import vllm.model_executor.models.registry as _reg

    def _aurora_run_in_subprocess(fn):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "registry_output.tmp")
            import cloudpickle  # already a vllm dependency

            data = cloudpickle.dumps((fn, out_file))

            result = subprocess.run(
                _reg._SUBPROCESS_COMMAND,
                input=data,
                capture_output=True,
            )
            try:
                result.check_returncode()
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Aurora registry subprocess failed "
                    f"(exit {exc.returncode}):\n{result.stderr.decode()}"
                ) from exc

            with open(out_file, "rb") as fh:
                return pickle.load(fh)

    # Patching the module-level name is sufficient: all internal callers inside
    # registry.py resolve names through globals(), so the patch is transparent.
    _reg._run_in_subprocess = _aurora_run_in_subprocess


_patch_registry_subprocess()

# Run the API server as __main__ so that its ``if __name__ == '__main__':``
# block executes normally.  alter_sys=True updates sys.argv[0] to the module
# file path, preserving the remaining argv entries for argparse.
import runpy

runpy.run_module(
    "vllm.entrypoints.openai.api_server",
    run_name="__main__",
    alter_sys=True,
)
