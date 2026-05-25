#!/usr/bin/env python3
"""Aurora vLLM entrypoint.

Patches vLLM's ``_run_in_subprocess`` to set ``ONEAPI_DEVICE_SELECTOR=cpu``
for the model-registry subprocess *before* the API server starts.

Root cause
----------
On Aurora (Intel PVC / XPU), vLLM's model-registry inspection spawns a
subprocess via ``subprocess.run([sys.executable, '-m',
'vllm.model_executor.models.registry'], ...)`` **without** mpiexec.  That
child process is NOT a PALS-managed rank and therefore lacks the Level Zero
device-fabric permissions granted by PALS to mpiexec-launched ranks.  When
the child tries to initialise the XPU (triggered by importing IPEX at
``vllm.model_executor.models.mistral`` module level), Level Zero fails and
the process receives SIGSEGV.

Fix
---
The registry subprocess only needs to introspect Python class attributes
(``supports_multimodal``, ``is_text_generation_model``, etc.) — it never
runs any GPU kernels.  Setting ``ONEAPI_DEVICE_SELECTOR=cpu`` in the child's
environment avoids Level Zero initialisation entirely, so PALS permissions
are irrelevant and the subprocess completes without crashing.

This entrypoint must be launched *instead of*
``python -m vllm.entrypoints.openai.api_server``::

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

            env = os.environ.copy()
            # Model-registry subprocess only inspects class attributes;
            # it never needs an XPU device.  cpu selector avoids Level Zero
            # initialisation and the SIGSEGV that follows in non-PALS processes.
            env["ONEAPI_DEVICE_SELECTOR"] = "cpu"

            result = subprocess.run(
                _reg._SUBPROCESS_COMMAND,
                input=data,
                capture_output=True,
                env=env,
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
