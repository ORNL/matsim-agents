#!/usr/bin/env python3
"""Aurora vLLM entrypoint.

Thin wrapper that patches vLLM's ``_run_in_subprocess`` to set
``ONEAPI_DEVICE_SELECTOR=cpu:*`` for the model-registry subprocess, then
runs the API server.

Root cause
----------
On some Aurora (Intel PVC / XPU) compute nodes, vLLM's model-registry
subprocess — spawned via ``subprocess.run()`` as a plain fork+exec child of
the mpiexec-launched API server — crashes with SIGSEGV during Level Zero
device initialisation.  The subprocess is not an mpiexec/PALS rank and on
those nodes it lacks the device-fabric permissions required to open the GPU
context.  The crash is node-specific: on other nodes the plain subprocess
works without any workaround.

Fix
---
The registry subprocess only introspects Python class attributes
(``supports_multimodal``, ``is_text_generation_model``, etc.) and never
executes GPU kernels.  Setting ``ONEAPI_DEVICE_SELECTOR=cpu:*`` in its
environment prevents IPEX/SYCL from attempting Level Zero initialisation
altogether.  This is a no-op on nodes where the plain subprocess already
succeeds.

Usage::

    mpiexec -n 1 --ppn 1 \\
      env -u PMI_RANK ... \\
      python aurora_vllm_entrypoint.py [api_server_args...]
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
            # The registry subprocess only inspects class attributes — it never
            # runs GPU kernels.  On some Aurora compute nodes, the subprocess
            # (a plain fork+exec child, not an mpiexec rank) crashes with SIGSEGV
            # when importing IPEX triggers Level Zero initialisation without full
            # PALS device-fabric permissions.  Setting cpu:* avoids Level Zero
            # init entirely.  On nodes where the plain subprocess already works,
            # this selector is harmless (no GPU init is skipped at runtime).
            # Note: 'cpu' and 'cpu:*' are rejected by some node SYCL versions;
            # 'opencl:cpu' is the canonical backend:device_type form and works
            # consistently across Aurora's driver versions.
            env["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"

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
